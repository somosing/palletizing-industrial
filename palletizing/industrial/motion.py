"""Joint-space RRT-Connect with a separate Bullet collision world.

Collision sampling is discrete (configured angular resolution), not a proof of
continuous collision freedom. Carried cartons are checked through the path.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from ..backends.bullet import xyzw, wxyz
from ..kinematics import rotation_error


class MotionError(RuntimeError):
    pass


class CollisionPlanner:
    def __init__(self, scene, robot, config):
        from pybullet_utils.bullet_client import BulletClient
        import pybullet as pb
        self.scene, self.robot, self.c = scene, robot, config
        self.p = BulletClient(connection_mode=pb.DIRECT)
        self.body = self.p.loadURDF(str(scene.asset), scene.base_position,
                                   useFixedBase=True, flags=pb.URDF_USE_INERTIA_FROM_FILE)
        self.rng = np.random.default_rng(config['simulation']['seed'])
        self.obstacles = {}
        self.obstacle_sizes = {}
        self.held = None
        self.relative = None
        self.last_reason = ''
        self.checks = 0
        self.plans = []
        self.resolution = config['motion']['collision_resolution_rad']
        self.clearance = config['motion']['clearance_m']
        self.pairs = self._self_pairs()

    def _self_pairs(self):
        p, body = self.p, self.body
        groups = {-1: -1}
        parents = {}
        links = []
        for i in range(p.getNumJoints(body)):
            info = p.getJointInfo(body,i)
            parent = info[16]
            if info[2] == p.JOINT_FIXED:
                groups[i] = groups[parent]
            else:
                groups[i] = i
                parents[i] = groups[parent]
            if p.getCollisionShapeData(body,i):
                links.append(i)
        pairs = []
        for a in links:
            for b in links:
                ga, gb = groups[a], groups[b]
                if a < b and ga != gb and parents.get(ga) != gb and parents.get(gb) != ga:
                    pairs.append((a,b))
        return pairs

    def sync(self, gripper=None):
        p = self.p
        for body, spec in self.scene.collision_objects.items():
            if body in self.scene.tracked_cartons:
                spec,pos,q=self.scene.tracked_cartons[body]
            else:
                # Only static CAD obstacles use their nominal simulator placement.
                pos,q=self.scene.p.getBasePositionAndOrientation(body)
            if body not in self.obstacles or not np.allclose(self.obstacle_sizes[body],spec):
                if body in self.obstacles:p.removeBody(self.obstacles[body])
                shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=np.array(spec)/2)
                self.obstacles[body]=p.createMultiBody(0,shape)
                self.obstacle_sizes[body]=np.array(spec)
            p.resetBasePositionAndOrientation(self.obstacles[body],pos,q)
        self.held = None if gripper is None else gripper.body
        self.relative = None if gripper is None else gripper.planning_relative

    def _ik_seeds(self, reference, attempts):
        """Build deterministic nearby elbow branches before broad random seeds."""
        reference = np.asarray(reference, dtype=float)
        attempts = max(1, int(attempts))
        seeds = [reference.copy()]
        # UR arms have redundant elbow/wrist configurations. Small structured
        # shoulder/elbow perturbations are much more useful than only two broad
        # random guesses for a vertical lift near the robot base.
        variations = [
            (2, .25), (2, -.25), (2, .50), (2, -.50),
            (1, .20), (1, -.20), (1, .40), (1, -.40),
            (1, .25, 2, -.35), (1, -.25, 2, .35),
            (0, .25, 2, .25), (0, -.25, 2, -.25),
            (3, .35), (3, -.35), (4, .35), (4, -.35),
        ]
        for variation in variations:
            if len(seeds) >= attempts:
                break
            candidate = reference.copy()
            candidate[variation[0]] += variation[1]
            if len(variation) == 4:
                candidate[variation[2]] += variation[3]
            candidate = np.clip(candidate, self.robot.low + 1e-5, self.robot.high - 1e-5)
            if not any(np.allclose(candidate, existing, atol=1e-5) for existing in seeds):
                seeds.append(candidate)
        while len(seeds) < attempts:
            seeds.append(self.rng.uniform(self.robot.low, self.robot.high))
        return seeds

    def set_q(self, q):
        for j,value in zip(self.robot.joints,q):
            self.p.resetJointState(self.body,j,float(value))
        if self.held is not None:
            link = self.p.getLinkState(self.body,self.robot.wrist,computeForwardKinematics=True)
            pose = self.p.multiplyTransforms(link[4],link[5],*self.relative)
            self.p.resetBasePositionAndOrientation(self.obstacles[self.held],*pose)

    def valid(self, q):
        self.checks += 1
        if np.any(q < self.robot.low) or np.any(q > self.robot.high) or not np.isfinite(q).all():
            self.last_reason = 'joint limits'
            return False
        self.set_q(q)
        p, body = self.p, self.body
        for a,b in self.pairs:
            if p.getClosestPoints(body,body,self.clearance,linkIndexA=a,linkIndexB=b):
                self.last_reason = f'self collision links {a},{b}'
                return False
        for original, other in self.obstacles.items():
            for contact in p.getClosestPoints(body,other,self.clearance):
                link = contact[3]
                if original == self.held and link in self.robot.tool_links:
                    continue
                # Fixed mounting links are intentionally seated on the pedestal.
                if original == self.scene.pedestal and link in self.robot.mount_links:
                    continue
                self.last_reason = f'robot link {link} vs object {original}'
                return False
        if self.held is not None:
            held = self.obstacles[self.held]
            _, quat = p.getBasePositionAndOrientation(held)
            normal = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)[:,2]
            if normal[2] < np.cos(self.c['motion']['max_carry_tilt_rad']):
                self.last_reason = 'carried carton tilt'
                return False
            for original, other in self.obstacles.items():
                if original == self.held:
                    continue
                # Resting face contact is allowed; penetration is not.
                if any(pt[8] < -0.0015 for pt in p.getClosestPoints(held,other,0.0)):
                    self.last_reason = f'carried carton vs object {original}'
                    return False
        return True

    def edge(self, a, b):
        steps = max(1,int(np.ceil(np.max(np.abs(b-a))/self.resolution)))
        return all(self.valid(a+(b-a)*t) for t in np.linspace(0,1,steps+1))

    def ik(self, pos, quat, reference, attempts=80):
        candidates = []
        diagnostics = []
        seeds = self._ik_seeds(reference, attempts)
        for seed in seeds:
            self.set_q(seed)
            q = np.asarray(self.p.calculateInverseKinematics(
                self.body,self.robot.wrist,pos,xyzw(quat),
                lowerLimits=self.robot.low.tolist(),upperLimits=self.robot.high.tolist(),
                jointRanges=(self.robot.high-self.robot.low).tolist(),restPoses=seed.tolist(),
                maxNumIterations=600,residualThreshold=1e-6))
            # Choose equivalent revolute angles closest to the current configuration.
            q += 2*np.pi*np.round((reference-q)/(2*np.pi))
            self.set_q(q)
            link = self.p.getLinkState(self.body,self.robot.wrist,computeForwardKinematics=True)
            diagnostics.append((float(np.linalg.norm(np.array(link[4])-pos)), float(np.linalg.norm(rotation_error(quat,wxyz(link[5]))))))
            if (np.linalg.norm(np.array(link[4])-pos) < 0.003
                and np.linalg.norm(rotation_error(quat,wxyz(link[5]))) < 0.025
                and self.valid(q)):
                candidates.append(q)
                if len(candidates)>=4:
                    break
        if not candidates:
            self.last_reason = f"target={np.asarray(pos).tolist()}, best pose residual={min(diagnostics)}, last collision={self.last_reason}"
        return sorted(candidates,key=lambda q: np.linalg.norm(q-reference))

    def connect(self, start, goal):
        if not self.valid(start) or not self.valid(goal):
            raise MotionError('Invalid planning endpoint: '+self.last_reason)
        if self.edge(start,goal):
            return [start,goal]
        trees = [([start],[-1]),([goal],[-1])]
        deadline = time.monotonic()+self.c['motion']['planning_timeout_s']
        step = self.c['motion']['rrt_step_rad']
        def extend(tree, target):
            nodes, parents = tree
            near = int(np.argmin([np.linalg.norm(q-target) for q in nodes]))
            delta = target-nodes[near]
            new = nodes[near]+delta*min(1.0,step/max(np.linalg.norm(delta),1e-12))
            if not self.edge(nodes[near],new):
                return None, False
            nodes.append(new); parents.append(near)
            return len(nodes)-1, np.linalg.norm(new-target)<1e-8
        def trace(tree, index):
            out=[]
            while index!=-1:
                out.append(tree[0][index]); index=tree[1][index]
            return out[::-1]
        for iteration in range(self.c['motion']['rrt_iterations']):
            if time.monotonic()>deadline:
                break
            side = iteration%2
            sample = trees[1-side][0][0] if self.rng.random()<0.15 else self.rng.uniform(self.robot.low,self.robot.high)
            a,_ = extend(trees[side],sample)
            if a is None:
                continue
            target = trees[side][0][a]
            for _ in range(100):
                if time.monotonic()>deadline:
                    break
                b,reached = extend(trees[1-side],target)
                if b is None:
                    break
                if reached:
                    one, two = trace(trees[side],a), trace(trees[1-side],b)
                    path = one+two[-2::-1]
                    if side==1:
                        path=path[::-1]
                    # Deterministic shortcutting; every shortcut is collision checked.
                    out=[path[0]]; i=0
                    while i<len(path)-1:
                        j=len(path)-1
                        while j>i+1 and not self.edge(path[i],path[j]):
                            j-=1
                        out.append(path[j]); i=j
                    return out
        raise MotionError('RRT exhausted search budget; no validated path')

    def plan(self, pos, quat, start, cartesian=False):
        before, begun = self.checks,time.perf_counter()
        if cartesian:
            self.set_q(start)
            link = self.p.getLinkState(self.body,self.robot.wrist,computeForwardKinematics=True)
            origin = np.array(link[4]); q=start.copy(); path=[q]
            orientations = Slerp([0,1],Rotation.from_quat([link[5],xyzw(quat)]))
            angle = np.linalg.norm(rotation_error(quat,wxyz(link[5])))
            n=max(2,int(np.ceil(np.linalg.norm(np.array(pos)-origin)/0.015)),int(np.ceil(angle/.08)))
            for t in np.linspace(0,1,n+1)[1:]:
                intermediate=wxyz(orientations(t).as_quat())
                attempts = int(self.c['motion'].get('cartesian_ik_attempts', 12))
                options = self.ik(origin+t*(np.array(pos)-origin),intermediate,q,
                                  attempts=attempts)
                good = next((candidate for candidate in options if self.edge(q,candidate)),None)
                if good is None:
                    raise MotionError('No collision-free Cartesian approach/retreat: '+self.last_reason)
                path.append(good); q=good
        else:
            options = self.ik(np.asarray(pos),quat,start)
            if not options:
                raise MotionError('No collision-free IK solution: '+self.last_reason)
            path=None
            for goal in options:
                try:
                    path=self.connect(start,goal);break
                except MotionError:
                    continue
            if path is None:
                raise MotionError('No collision-free path to available IK branches')
        self.plans.append({'seconds':time.perf_counter()-begun,'collision_checks':self.checks-before,
                           'waypoints':len(path),'cartesian':cartesian})
        return path

    def close(self):
        self.p.disconnect()


class URController:
    def __init__(self, scene, config):
        self.scene,self.p,self.robot,self.c = scene,scene.p,scene.robot,config
        p=self.p
        infos=[p.getJointInfo(self.robot,i) for i in range(p.getNumJoints(self.robot))]
        self.joints=[i for i,info in enumerate(infos) if info[2]==p.JOINT_REVOLUTE]
        names={info[12].decode():i for i,info in enumerate(infos)}
        self.wrist=names['tool0']
        self.tool_links={names[n] for n in ['tool0','flange','wrist_3_link','vacuum_tool'] if n in names}
        self.mount_links={-1,*[names[n] for n in ['base_link','base_link_inertia'] if n in names]}
        self.low=np.array([infos[j][8]+0.02 for j in self.joints])
        self.high=np.array([infos[j][9]-0.02 for j in self.joints])
        self.effort=np.array([infos[j][10] for j in self.joints])
        self.speed=np.minimum([infos[j][11] for j in self.joints],config['robot']['max_joint_speed'])
        self.planner=CollisionPlanner(scene,self,config)
        self.gripper=None

    def state(self):
        s=self.p.getJointStates(self.robot,self.joints)
        return np.array([x[0] for x in s]),np.array([x[1] for x in s])

    def pose(self):
        s=self.p.getLinkState(self.robot,self.wrist,computeForwardKinematics=True)
        return np.array(s[4]),wxyz(s[5])

    def command(self,q,velocity=None):
        self.p.setJointMotorControlArray(self.robot,self.joints,self.p.POSITION_CONTROL,
            targetPositions=np.asarray(q).tolist(),targetVelocities=[0.0]*6 if velocity is None else velocity.tolist(),
            forces=self.effort.tolist(),positionGains=[0.3]*6,velocityGains=[1.0]*6)

    def hold(self):
        self.command(self.state()[0])

    def bootstrap(self):
        seed=np.array([0,-1.5,1.8,-1.8,-1.57,0.0])
        self.planner.sync()
        options=self.planner.ik(np.asarray(self.c['robot']['scan_position']),
                                self.c['robot']['scan_orientation'],seed,attempts=80)
        if not options:
            raise MotionError('UR10e scan pose is unreachable or colliding: '+self.planner.last_reason)
        for j,q in zip(self.joints,options[0]):
            self.p.resetJointState(self.robot,j,float(q))
        self.command(options[0])

    def move(self,position,quaternion,step,cartesian=False):
        self.planner.sync(self.gripper)
        path=self.planner.plan(position,quaternion,self.state()[0],cartesian)
        self.execute(path,position,quaternion,step)

    def execute(self,path,position,quaternion,step):
        dt=1/self.c['simulation']['physics_hz']
        for a,b in zip(path[:-1],path[1:]):
            delta=b-a
            duration=max(0.12,float(np.max(1.875*np.abs(delta)/self.speed)),
                         float(np.max(np.sqrt(5.774*np.abs(delta)/self.c['robot']['max_joint_acceleration']))))
            count=int(np.ceil(duration/dt));duration=count*dt
            for k in range(1,count+1):
                t=k/count; blend=10*t**3-15*t**4+6*t**5
                derivative=(30*t**2-60*t**3+30*t**4)/duration
                self.command(a+delta*blend,delta*derivative)
                step()
                if np.max(np.abs(self.state()[0]-(a+delta*blend)))>0.15:
                    raise MotionError('Joint trajectory tracking error exceeded 0.15 rad')
        self.command(path[-1])
        for _ in range(int(2/dt)):
            step()
            pos,q=self.pose()
            if (np.linalg.norm(pos-position)<self.c['robot']['position_tolerance']
                and np.linalg.norm(rotation_error(quaternion,q))<self.c['robot']['orientation_tolerance']
                and np.max(np.abs(self.state()[1]))<0.02):
                return
        raise MotionError('Final Cartesian target did not settle within tolerance')

    def transport(self, position, quaternion, step):
        """Prevalidate an upright Cartesian transfer, with alternate elevated arcs.

        Empty-arm moves use RRT-Connect. Loaded transfers use this orientation-
        preserving task-space search to avoid tipping cartons during random search.
        """
        self.planner.sync(self.gripper)
        start=self.state()[0]; origin,ori=self.pose()
        position=np.asarray(position,float)
        a0,a1=np.arctan2(origin[1],origin[0]),np.arctan2(position[1],position[0])
        delta=(a1-a0+np.pi)%(2*np.pi)-np.pi
        r0,r1=np.linalg.norm(origin[:2]),np.linalg.norm(position[:2])
        candidates=[[(position,quaternion)]]
        for extra in [0.0,.10,.20]:
            arc=[];z=max(origin[2],position[2])+extra
            if extra:arc.append((np.r_[origin[:2],z],ori))
            interp=Slerp([0,1],Rotation.from_quat([xyzw(ori),xyzw(quaternion)]))
            for t in np.linspace(0,1,9)[1:]:
                angle=a0+delta*t;radius=(1-t)*r0+t*r1
                arc.append((np.array([radius*np.cos(angle),radius*np.sin(angle),z]),wxyz(interp(t).as_quat())))
            if extra:arc.append((position,quaternion))
            candidates.append(arc)
        errors=[]
        for route in candidates:
            q=start.copy();path=[q]
            try:
                for pos,ori in route:
                    part=self.planner.plan(pos,ori,q,cartesian=True)
                    path.extend(part[1:]);q=part[-1]
                self.execute(path,position,quaternion,step)
                return
            except MotionError as exc:
                # Only planning errors may try another path. Execution errors propagate.
                if np.max(np.abs(self.state()[0]-start))>.01:
                    raise
                errors.append(str(exc))
        raise MotionError('No upright payload transfer found: '+errors[-1])
