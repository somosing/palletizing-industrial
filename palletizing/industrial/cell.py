"""Perception-driven orchestration and physical verification for the UR10e cell."""
import json
import logging
import time
from pathlib import Path
import numpy as np
from ..backends.bullet import Scene as LegacyScene, WristCamera, EyeInHandPerception, xyzw, wxyz
from ..kinematics import DOWN, multiply, yaw_quaternion, rotation
from .motion import URController, MotionError
from .packing import SupportPacker, PalletFull
from .online_planner import plan_online_pick, reconcile_first_placement
from .physical_conveyor import FiniteConveyorBuffer, make_arrival_times

LOG=logging.getLogger(__name__)


def conveyor_interpolate(start,target,progress):
    """Smoothstep interpolation for visual infeed motion, clamped to [0, 1]."""
    start=np.asarray(start,float);target=np.asarray(target,float)
    if start.shape!=target.shape or not np.isfinite(start).all() or not np.isfinite(target).all():
        raise ValueError('Conveyor endpoints must be finite arrays of equal shape')
    t=float(progress)
    if not np.isfinite(t):
        raise ValueError('Conveyor progress must be finite')
    t=float(np.clip(t,0.0,1.0))
    blend=t*t*(3.0-2.0*t)
    return start+(target-start)*blend


class CellScene(LegacyScene):
    def __init__(self,p,c,root,inventory,obstacle=False):
        self.p,self.c,self.inventory=p,c,inventory
        self.collision_objects={}
        self.tracked_cartons={}
        self.rng=np.random.default_rng(c['simulation']['seed'])
        self.boxes=[];self.box_dimensions=[];self.verified={};self.spawned=set()
        self.infeed_motions={}
        self.pallet_carton_by_placement={}
        self.load_plan=None
        p.setRealTimeSimulation(0);p.setGravity(0,0,-9.81)
        p.setTimeStep(1/c['simulation']['physics_hz'])
        p.setPhysicsEngineParameter(numSolverIterations=150,deterministicOverlappingPairs=1)
        self.floor=self.box([8,5,.1],[0,0,-.05],[.85,.87,.89,1])
        self.surfaces={}
        for name,color in [('table',[.28,.32,.36,1]),('pallet',[.5,.32,.14,1])]:
            d=c[name]['dimensions']
            self.surfaces[name]=self.box(d,[*c[name]['center_xy'],d[2]/2],color)
        self._build_conveyor()
        self.base_position=c['robot']['base_position']
        h=self.base_position[2]
        self.pedestal=self.box([.24,.24,h],[0,0,h/2],[.18,.2,.23,1])
        if obstacle:
            # Guard post between pick and pallet. Planner must account for it.
            self.box([.13,.13,.70],[.48,.30,.35],[.8,.35,.08,1])
        self.asset=root/c['robot']['asset_relative_path']
        self.robot=p.loadURDF(str(self.asset),self.base_position,useFixedBase=True,
                 flags=p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_SELF_COLLISION|p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        p.resetDebugVisualizerCamera(5.0,52,-27,[1.35,.10,.48])

    def _visual_box(self,dimensions,position,color):
        """Add a render-only conveyor detail without changing collision planning."""
        p=self.p
        visual=p.createVisualShape(p.GEOM_BOX,halfExtents=np.asarray(dimensions,float)/2,
                                   rgbaColor=color)
        return p.createMultiBody(baseMass=0,baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=visual,basePosition=position)

    def _build_conveyor(self):
        cfg=self.c.get('conveyor',{})
        self.conveyor_enabled=bool(cfg.get('enabled',True))
        table=np.asarray(self.c['table']['dimensions'],float)
        self.conveyor_top_z=float(table[2])
        center=np.asarray(self.c['table']['center_xy'],float)
        self.conveyor_start_x=float(center[0]+cfg.get('feed_start_offset_x',.78))
        self.conveyor_duration=float(cfg.get('feed_duration_s',3.0))
        self.buffer_first_slot_x=float(center[0]+cfg.get('buffer_first_slot_offset_x',.50))
        self.buffer_slot_pitch=float(cfg.get('buffer_slot_pitch_m',.47))
        self.pick_station_xy=center.copy()
        self.conveyor_visuals=[]
        if not self.conveyor_enabled:
            return
        length=float(cfg.get('length',1.45));width=float(cfg.get('width',.52))
        thickness=float(cfg.get('deck_thickness',.024))
        belt_center=np.r_[center[0]+float(cfg.get('center_offset_x',.35)),center[1]]
        deck=[length,width,thickness]
        deck_pos=[*belt_center,self.conveyor_top_z-thickness/2]
        # This collidable deck extends the existing pick table into an infeed
        # belt. Keep it in the collision world so the robot plans around it.
        body=LegacyScene.box(self,deck,deck_pos,[.075,.09,.10,1])
        self.collision_objects[body]=deck
        rail_color=[.22,.25,.28,1]
        rail_length=length
        rail_height=.075
        rail_y=width/2+.012
        for sign in (-1,1):
            y=float(belt_center[1]+sign*rail_y)
            rail=[rail_length,.024,rail_height]
            rail_body=self.box(rail,[belt_center[0],y,self.conveyor_top_z+rail_height/2],rail_color)
            self.conveyor_visuals.append(rail_body)
        count=int(cfg.get('roller_count',12))
        roller_width=width-.055
        top=self.conveyor_top_z+.004
        for index in range(count):
            x=float(belt_center[0]-length/2+(index+.5)*length/count)
            visual=[.018,roller_width,.006]
            pos=[x,float(belt_center[1]),top]
            self.conveyor_visuals.append(
                self._visual_box(visual,pos,[.48,.51,.54,1]))
        # Render-only gantry marks the selectable accumulation buffer/shuttle.
        # The shuttle path is elevated above cartons and never moves a carton
        # through another carton or asks the arm to pick outside its scan zone.
        gantry_z=self.conveyor_top_z+.60
        for sign in (-1,1):
            y=float(belt_center[1]+sign*(width/2-.025))
            self.conveyor_visuals.append(self._visual_box(
                [.035,length,.035],[belt_center[0],y,gantry_z],[.92,.43,.08,1]))
        for x in (belt_center[0]-length/2+.04,belt_center[0]+length/2-.04):
            for sign in (-1,1):
                y=float(belt_center[1]+sign*(width/2-.025))
                self.conveyor_visuals.append(self._visual_box(
                    [.035,.035,gantry_z-self.conveyor_top_z],
                    [x,y,(gantry_z+self.conveyor_top_z)/2],[.35,.37,.40,1]))

    def box(self,d,pos,color,mass=0,yaw=0):
        body=super().box(d,pos,color,mass,yaw)
        self.collision_objects[body]=list(d)
        return body

    def buffer_slot_xy(self,slot):
        if not isinstance(slot,int) or not 0<=slot<int(self.c['conveyor'].get('buffer_capacity',4)):
            raise ValueError('Invalid conveyor buffer slot')
        center=np.asarray(self.c['table']['center_xy'],float)
        return np.array([self.buffer_first_slot_x+slot*self.buffer_slot_pitch,center[1]])

    def feed(self,index,slot=0,now=0.0):
        if index in self.spawned or index!=len(self.boxes):
            raise RuntimeError('Inventory duplicated or fed out of order')
        item=self.inventory[index];d=np.array(item['dimensions'])
        slot_xy=self.buffer_slot_xy(slot) if self.conveyor_enabled else (
            np.asarray(self.c['table']['center_xy'])+self.rng.uniform(-.025,.025,2))
        start_xy=np.array([self.conveyor_start_x,slot_xy[1]]) if self.conveyor_enabled else slot_xy
        yaw=float(self.rng.uniform(-.35,.35))
        body=self.box(d,[*start_xy,self.conveyor_top_z+d[2]/2+.002],
                      self.rng.uniform([.55,.30,.10],[.82,.62,.28]).tolist()+[1],item['mass'],
                      yaw)
        self.boxes.append(body);self.box_dimensions.append(d);self.spawned.add(index)
        # Before perception, protect the whole possible indexed-infeed envelope.
        envelope=np.r_[np.sqrt(2)*np.array(self.c['box']['maximum_dimensions'][:2])+.06,
                       self.c['box']['maximum_dimensions'][2]]
        self.tracked_cartons[body]=(envelope,[*start_xy,self.conveyor_top_z+envelope[2]/2],
                                    self.p.getQuaternionFromEuler([0,0,yaw]))
        if self.conveyor_enabled:
            self.infeed_motions[index]={
                'slot':slot,'start_time':float(now),
                'duration':self.conveyor_duration,
                'source':np.array([*start_xy,self.conveyor_top_z+float(d[2])/2+.002]),
                'target':np.array([*slot_xy,self.conveyor_top_z+float(d[2])/2+.002]),
                'orientation':self.p.getQuaternionFromEuler([0,0,yaw]),
                'envelope':envelope}
        return body

    def advance_infeed(self,now):
        """Animate due cartons up and along the belt's clear selector path."""
        completed=[];p=self.p
        for index,motion in list(self.infeed_motions.items()):
            progress=float(np.clip((float(now)-motion['start_time'])/motion['duration'],0,1))
            source=motion['source'];target=motion['target']
            clearance=max(source[2],target[2])+float(self.c['conveyor'].get('shuttle_clearance_m',.32))
            waypoints=[source,np.r_[source[:2],clearance],np.r_[target[:2],clearance],target]
            scaled=progress*(len(waypoints)-1)
            segment=min(int(scaled),len(waypoints)-2)
            local=1.0 if progress>=1 else scaled-segment
            position=conveyor_interpolate(waypoints[segment],waypoints[segment+1],local)
            body=self.boxes[index]
            p.resetBasePositionAndOrientation(body,position.tolist(),motion['orientation'])
            p.resetBaseVelocity(body,[0,0,0],[0,0,0])
            self.tracked_cartons[body]=(motion['envelope'],position.tolist(),motion['orientation'])
            if progress>=1:
                p.resetBasePositionAndOrientation(body,target.tolist(),motion['orientation'])
                completed.append(index)
                del self.infeed_motions[index]
        return completed

    def shuttle_to_pick(self,index,slot,step):
        """Move one measured carton through the clear overhead selector path."""
        if not self.conveyor_enabled:
            return
        p=self.p;body=self.boxes[index]
        source=np.asarray(p.getBasePositionAndOrientation(body)[0],float)
        orientation=p.getBasePositionAndOrientation(body)[1]
        dims=self.box_dimensions[index]
        target_xy=np.asarray(self.c['table']['center_xy'],float)
        target_z=self.conveyor_top_z+float(dims[2])/2+.002
        clearance=max(source[2],target_z)+float(self.c['conveyor'].get('shuttle_clearance_m',.32))
        waypoints=[np.r_[source[:2],clearance],np.r_[target_xy,clearance],
                   np.r_[target_xy,target_z]]
        current=source
        frames=max(12,int(round(float(self.c['conveyor'].get('shuttle_duration_s',.8))*
                                self.c['simulation']['physics_hz']/len(waypoints))))
        envelope=np.r_[np.sqrt(2)*np.asarray(self.c['box']['maximum_dimensions'][:2])+.06,
                       self.c['box']['maximum_dimensions'][2]]
        yaw_quat=orientation
        for target in waypoints:
            for frame in range(1,frames+1):
                position=conveyor_interpolate(current,target,frame/frames)
                p.resetBasePositionAndOrientation(body,position.tolist(),yaw_quat)
                p.resetBaseVelocity(body,[0,0,0],[0,0,0])
                self.tracked_cartons[body]=(envelope,position.tolist(),yaw_quat)
                step()
            current=target
        p.resetBasePositionAndOrientation(body,[*target_xy,target_z],yaw_quat)
        p.resetBaseVelocity(body,[0,0,0],[0,0,0])
        self.tracked_cartons[body]=(envelope,[*target_xy,target_z],yaw_quat)

    def record_pallet_placement(self,placement_index,carton_index):
        if not hasattr(self,'pallet_carton_by_placement'):
            self.pallet_carton_by_placement={}
        if placement_index in self.pallet_carton_by_placement:
            raise RuntimeError('Duplicate pallet placement identity')
        self.pallet_carton_by_placement[int(placement_index)]=int(carton_index)

    def animate_feed(self,index,step):
        """Advance the carton from belt entry to the wrist-camera pick zone."""
        if not self.conveyor_enabled:
            return
        body=self.boxes[index];p=self.p
        start,_=p.getBasePositionAndOrientation(body)
        target_xy=np.asarray(self.c['table']['center_xy'],float)+self.rng.uniform(-.025,.025,2)
        target=np.array([target_xy[0],target_xy[1],self.conveyor_top_z+self.box_dimensions[index][2]/2+.002])
        source=np.asarray(start,float)
        _,orientation=p.getBasePositionAndOrientation(body)
        frames=max(1,int(round(self.conveyor_duration*self.c['simulation']['physics_hz'])))
        yaw=float(p.getEulerFromQuaternion(orientation)[2])
        for frame in range(1,frames+1):
            linear=frame/frames
            position=conveyor_interpolate(source,target,linear)
            q=p.getQuaternionFromEuler([0,0,yaw])
            p.resetBasePositionAndOrientation(body,position.tolist(),q)
            p.resetBaseVelocity(body,[0,0,0],[0,0,0])
            envelope=np.r_[np.sqrt(2)*np.array(self.c['box']['maximum_dimensions'][:2])+.06,
                           self.c['box']['maximum_dimensions'][2]]
            conservative_center=[position[0],position[1],self.conveyor_top_z+envelope[2]/2]
            self.tracked_cartons[body]=(envelope,conservative_center,q)
            step()
        # Land exactly at the camera's pick-zone target and leave normal physics
        # enabled for settling and the subsequent measured pick.
        p.resetBasePositionAndOrientation(body,target.tolist(),p.getQuaternionFromEuler([0,0,yaw]))
        p.resetBaseVelocity(body,[0,0,0],[0,0,0])

    def track(self,index,dimensions,position,yaw):
        # Four millimetres of lateral uncertainty padding; no simulated box pose read.
        d=np.asarray(dimensions,float)+[.008,.008,0]
        self.tracked_cartons[self.boxes[index]]=(d,list(position),self.p.getQuaternionFromEuler([0,0,float(yaw)]))

    def verify_placement(self,index,placement):
        body=self.boxes[index];p=self.p
        pos,q=p.getBasePositionAndOrientation(body);v,w=p.getBaseVelocity(body)
        tilt=np.arccos(np.clip(np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)[2,2],-1,1))
        error=float(np.linalg.norm(np.array(pos)-placement.center))
        # Grasp may preserve the carton yaw modulo pi; rectangles permit that symmetry.
        yaw=p.getEulerFromQuaternion(q)[2]
        yaw_error=abs(float(np.angle(np.exp(2j*(yaw-placement.yaw)))/2))
        # Perception can choose swapped local x/y dimensions, so use footprint mismatch too.
        actual_d=self.box_dimensions[index]
        R=np.abs(np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)[:2,:2])
        footprint=R@actual_d[:2]
        footprint_error=float(np.max(np.abs(footprint-np.array(placement.dimensions[:2]))))
        if placement.support is None:
            support=self.surfaces['pallet']
        else:
            support_index=self.pallet_carton_by_placement.get(int(placement.support))
            if support_index is None:
                raise RuntimeError('Pallet support carton identity is missing')
            support=self.boxes[support_index]
        supported=bool(p.getContactPoints(body,support))
        good=(error<self.c['process']['placement_tolerance'] and tilt<.06 and
              np.linalg.norm(v)<.03 and np.linalg.norm(w)<.1 and supported and footprint_error<.025)
        record={'index':index,'center_error_m':error,'tilt_rad':float(tilt),'yaw_error_rad':yaw_error,
                'footprint_error_m':footprint_error,'supported':supported,'position':list(pos),'verified':bool(good)}
        self.verified[index]=record
        if not good:
            raise RuntimeError('Placement verification failed: '+json.dumps(record))
        return record


class NoisyCamera(WristCamera):
    def __init__(self,p,robot,c):
        super().__init__(p,robot,c)
        self.noise=c['sensor'];self.rng=np.random.default_rng(c['simulation']['seed']+10000)
        self.blanked=False

    def capture(self,labels=False,**lighting):
        super().capture(labels,**lighting)
        near,far=self.c['clipping_range']
        z=super().get_depth()
        valid=(z>near)&(z<far*.98)
        z[valid]+=self.rng.normal(0,self.noise['depth_noise_std_m'],valid.sum())
        z=np.clip(z,near,far)
        self.depth=far*(z-near)/((far-near)*z)
        self.depth[self.rng.random(z.shape)<self.noise['depth_dropout']]=np.nan
        if self.blanked:
            self.depth[:]=np.nan


class Vacuum:
    def __init__(self,scene,robot):
        self.scene,self.robot,self.p=scene,robot,scene.p
        self.body=self.constraint=self.relative=None
        self.peak_force=self.peak_torque=0.0
        self.attached_at=None;self.lost=False;self.planning_relative=None

    def close(self,index,sim_time,estimate):
        if self.body is not None:
            raise RuntimeError('Vacuum already occupied')
        if self.scene.inventory[index]['mass']>self.scene.c['process']['max_tool_payload_kg']:
            raise RuntimeError('SKU mass exceeds configured tool payload')
        p=self.p;body=self.scene.boxes[index]
        pos,q=self.robot.pose();bp,bq=p.getBasePositionAndOrientation(body)
        tip=pos+rotation(q)@np.array([0,0,self.scene.c['robot']['tool_length']])
        top=np.array(bp)+[0,0,self.scene.box_dimensions[index][2]/2]
        if np.linalg.norm(tip-top)>.02 or (rotation(q)[:,2]@np.array(p.getMatrixFromQuaternion(bq)).reshape(3,3)[:,2])>-.985:
            raise RuntimeError('Vacuum seal geometry invalid')
        self.relative=p.multiplyTransforms(*p.invertTransform(pos,xyzw(q)),bp,bq)
        self.planning_relative=p.multiplyTransforms(*p.invertTransform(pos,xyzw(q)),estimate.position,p.getQuaternionFromEuler([0,0,estimate.yaw]))
        # Bullet constraint parent frame is expressed in the link COM frame.
        ls=p.getLinkState(self.robot.robot,self.robot.wrist,computeForwardKinematics=True)
        com_relative=p.multiplyTransforms(*p.invertTransform(ls[0],ls[1]),bp,bq)
        self.constraint=p.createConstraint(self.robot.robot,self.robot.wrist,body,-1,p.JOINT_FIXED,[0,0,0],com_relative[0],[0,0,0],parentFrameOrientation=com_relative[1])
        p.changeConstraint(self.constraint,maxForce=self.scene.c['process']['break_force'])
        self.body=body;self.attached_at=sim_time
        for link in self.robot.tool_links:
            p.setCollisionFilterPair(self.robot.robot,body,link,-1,0)

    def open(self):
        if self.constraint is None:
            raise RuntimeError('Cannot release empty vacuum')
        self.p.removeConstraint(self.constraint)
        for link in self.robot.tool_links:
            self.p.setCollisionFilterPair(self.robot.robot,self.body,link,-1,1)
        self.body=self.constraint=self.relative=self.planning_relative=None
        self.attached_at=None

    def check(self):
        p=self.p
        if self.lost:
            raise RuntimeError('Injected vacuum loss: cell stopped with unverified load')
        for contact in p.getContactPoints(bodyA=self.robot.robot):
            if contact[2]==self.scene.pedestal and contact[3] in self.robot.mount_links:
                continue
            if contact[8]<-.003 and contact[9]>1.0:
                raise RuntimeError(f'Unexpected robot contact: body={contact[2]}, link={contact[3]}, depth={contact[8]:.4f}')
        if self.constraint is None:
            return
        wrench=np.asarray(p.getConstraintState(self.constraint))
        force,torque=np.linalg.norm(wrench[:3]),np.linalg.norm(wrench[3:])
        self.peak_force=max(self.peak_force,float(force));self.peak_torque=max(self.peak_torque,float(torque))
        c=self.scene.c['process']
        if force>c['break_force'] or torque>c['break_torque']:
            raise RuntimeError(f'Vacuum wrench exceeded: {force:.2f} N, {torque:.2f} Nm')
        pos,q=self.robot.pose()
        expected=p.multiplyTransforms(pos,xyzw(q),*self.relative)
        actual=p.getBasePositionAndOrientation(self.body)
        if np.linalg.norm(np.array(expected[0])-actual[0])>.018:
            raise RuntimeError('Vacuum attachment drift exceeded 18 mm')


class IndustrialCell:
    def __init__(self,p,c,root,inventory,output,perception='depth',weights=None,device='0',
                 gui=False,speed=4.0,fault='none',obstacle=False,disturb_box=False):
        self.p,self.c,self.output=p,c,output
        self.gui,self.speed,self.fault=gui,speed,fault
        self.disturb_box=bool(disturb_box);self.disturbance_injected=False
        self.peak_joint_speed=np.zeros(6)
        self.time=0.0;self.state='INITIALIZE';self.index=0;self.events=[];self.observations=[]
        self.scene=CellScene(p,c,root,inventory,obstacle)
        self.robot=URController(self.scene,c)
        # Enable non-adjacent self collisions, excluding rigid/adjacent assemblies only.
        pairs=set(self.robot.planner.pairs)
        for a in range(-1,p.getNumJoints(self.scene.robot)):
            for b in range(a+1,p.getNumJoints(self.scene.robot)):
                p.setCollisionFilterPair(self.scene.robot,self.scene.robot,a,b,int((a,b) in pairs))
        self.robot.bootstrap()
        self.gripper=Vacuum(self.scene,self.robot);self.robot.gripper=self.gripper
        if perception=='learned':
            from ..learned_perception import LearnedEyeInHandPerception
            self.perception=LearnedEyeInHandPerception(p,self.robot,c,weights,device)
            self.perception.camera.source=NoisyCamera(p,self.robot,c)
            self.camera=self.perception.camera.source
        else:
            self.perception=EyeInHandPerception(p,self.robot,c)
            self.perception.known_dimensions=False
            self.perception.camera=NoisyCamera(p,self.robot,c)
            self.camera=self.perception.camera
        self.packer=SupportPacker(c['pallet']['center_xy'],c['packing']['usable_dimensions'],
                      c['pallet']['dimensions'][2],c['packing']['max_height'],c['packing']['gap'],
                      c['packing']['support_margin'],c['packing']['max_layers'],c['packing']['max_payload'],
                      c['packing']['min_support_fraction'],c['packing']['max_overhang'])
        conveyor_cfg=c.get('conveyor',{})
        self.arrival_times=make_arrival_times(
            len(inventory),float(conveyor_cfg.get('arrival_interval_s',2.5)),
            float(conveyor_cfg.get('arrival_jitter_fraction',.25)),
            int(c['simulation']['seed'])+3907)
        buffer_capacity=int(conveyor_cfg.get('buffer_capacity',4)) if self.scene.conveyor_enabled else 1
        self.conveyor_buffer=FiniteConveyorBuffer(self.arrival_times,buffer_capacity)
        self.planner_mode=conveyor_cfg.get('planner','beam-search')
        self.dimensioner_rng=np.random.default_rng(int(c['simulation']['seed'])+8803)
        self.planning_items=[None for _ in inventory]
        self.dimensioner_measurements=[]
        self.online_plans=[]
        self.online_placement_targets={}
        self.clock=time.perf_counter()

    def enter(self,state,**details):
        self.state=state
        event={'sim_seconds':self.time,'index':self.index,'state':state,**details}
        self.events.append(event)
        with (self.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        LOG.info('%s | verified=%d/%d',state,len(self.packer.placements),len(self.scene.inventory))

    def record_event(self,name,**details):
        event={'sim_seconds':self.time,'event':name,**details}
        self.events.append(event)
        with (self.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')

    def _advance_conveyor(self):
        for admission in self.conveyor_buffer.advance(self.time):
            carton=self.scene.inventory[admission.index]
            self.conveyor_buffer.mark_in_transit(admission.index)
            self.scene.feed(admission.index,admission.slot,self.time)
            # Simulated infeed dimensioner: noisy, timestamped measurements
            # arrive only once a carton reaches an admitted buffer slot.
            measured=np.asarray(carton['dimensions'],float)+self.dimensioner_rng.normal(0,.0015,3)
            measured=np.maximum(measured,.05)
            estimate=dict(carton)
            estimate['dimensions']=measured.tolist()
            self.planning_items[admission.index]=estimate
            measurement={'index':admission.index,'sku':carton.get('sku'),
                'true_dimensions_sim_only_m':[float(x) for x in carton['dimensions']],
                'measured_dimensions_m':measured.tolist(),
                'arrival_time_s':admission.arrival_time_s,
                'admitted_time_s':admission.admitted_time_s,
                'buffer_slot':admission.slot}
            self.dimensioner_measurements.append(measurement)
            self.record_event('CARTON_ARRIVED_INFEED',**measurement,
                buffer_occupancy=self.conveyor_buffer.occupancy,
                upstream_blocked_arrivals=self.conveyor_buffer.blocked_arrivals)
            LOG.info('CONVEYOR arrival %s -> slot %d | occupied=%d/%d blocked=%d',
                carton.get('sku',admission.index),admission.slot,
                self.conveyor_buffer.occupancy,self.conveyor_buffer.capacity,
                self.conveyor_buffer.blocked_arrivals)
        for index in self.scene.advance_infeed(self.time):
            self.conveyor_buffer.mark_ready(index)
            self.record_event('CARTON_BUFFER_READY',index=index,
                sku=self.scene.inventory[index].get('sku'),
                buffer_slot=self.conveyor_buffer.slot_for(index),
                buffer_occupancy=self.conveyor_buffer.occupancy)
            LOG.info('CONVEYOR carton %s reached buffer slot | ready=%d/%d',
                self.scene.inventory[index].get('sku'),len(self.conveyor_buffer.visible),
                self.conveyor_buffer.capacity)

    def step(self):
        if not self.p.isConnected():
            raise RuntimeError('Simulator closed before completion')
        self.p.stepSimulation();self.time+=1/self.c['simulation']['physics_hz']
        self._advance_conveyor()
        if self.time>self.c['simulation']['max_seconds']:
            raise RuntimeError('Whole-job simulation timeout')
        if (self.fault=='vacuum-loss' and self.gripper.constraint is not None and
            self.time-self.gripper.attached_at>.4):
            self.p.removeConstraint(self.gripper.constraint);self.gripper.constraint=None
            self.gripper.lost=True
        self.gripper.check()
        self.peak_joint_speed=np.maximum(self.peak_joint_speed,np.abs(self.robot.state()[1]))
        if self.gui:
            delay=self.clock+self.time/self.speed-time.perf_counter()
            if delay>0:
                time.sleep(min(delay,1/self.c['simulation']['physics_hz']/self.speed))

    def wait(self,seconds):
        for _ in range(int(seconds*self.c['simulation']['physics_hz'])):
            self.step()

    def move(self,pos,q=DOWN,cartesian=False):
        self.robot.move(np.asarray(pos,float),q,self.step,cartesian)

    def detect(self):
        for attempt in range(self.c['process']['detection_retries']+1):
            self.enter('DETECT',attempt=attempt)
            self.camera.blanked=self.fault=='detection-dropout' and self.index==0 and attempt==0
            self.perception.reset()
            for _ in range(8):
                self.wait(.05);self.camera.capture()
                pose=self.perception.detect()
                if pose is not None:
                    self.camera.blanked=False
                    return pose
            self.enter('RESCAN',reason=self.perception.last_error)
            self.camera.blanked=False
            self.wait(.2)
        raise RuntimeError('Detection exhausted retries: '+self.perception.last_error)

    def _online_carton_order(self):
        """Yield the next observed carton chosen from the physical buffer only."""
        count=len(self.scene.inventory)
        while len(self.packer.placements)<count:
            self._advance_conveyor()
            if not self.conveyor_buffer.visible:
                self.wait(1/self.c['simulation']['physics_hz'])
                continue
            visible=list(self.conveyor_buffer.visible)
            if self.planner_mode=='beam-search':
                depth=min(int(self.c['conveyor'].get('search_depth',4)),len(visible))
                try:
                    plan=plan_online_pick(
                        self.packer,self.planning_items,visible,visible,
                        beam_width=int(self.c['conveyor'].get('beam_width',24)),
                        search_depth=max(1,depth),
                        placement_branches=int(self.c['conveyor'].get('placement_branches',2)))
                except PalletFull as exc:
                    raise PalletFull(f'{exc}; visible={visible}, '
                        f'upstream={self.conveyor_buffer.upstream}, '
                        f'committed={len(self.packer.placements)}') from exc
                selected=int(plan.first_pick)
                planned_placement=(plan.placements[0] if plan.placements else None)
                if planned_placement is not None:
                    self.online_placement_targets[selected]=planned_placement
                details={'planner':'beam-search','sequence':[int(i) for i in plan.sequence],
                    'completed_depth':plan.completed_depth,'expanded_nodes':plan.expanded_nodes,
                    'search_score':list(plan.score),'optimality_certified':False,
                    'planned_first_placement':None if planned_placement is None else planned_placement.record()}
            else:
                feasible=[]
                for index in visible:
                    item=self.planning_items[index]
                    try:
                        candidate=self.packer.propose(item['dimensions'],item['mass'],item['capacity'])
                    except PalletFull:
                        continue
                    feasible.append((candidate.center[2],-candidate.support_fraction,
                                     -float(np.prod(item['dimensions'])),index))
                if not feasible:
                    raise PalletFull('No currently buffered carton can be placed safely; '
                        f'visible={visible}, upstream={self.conveyor_buffer.upstream}, '
                        f'committed={len(self.packer.placements)}')
                selected=int(min(feasible)[-1])
                details={'planner':'greedy-safe','sequence':[selected],
                         'completed_depth':1,'expanded_nodes':len(visible),
                         'optimality_certified':False}
            slot=self.conveyor_buffer.slot_for(selected)
            record={'selected_index':selected,'selected_sku':self.scene.inventory[selected]['sku'],
                'buffer_slot':slot,'visible_indices':visible,
                'upstream_indices':self.conveyor_buffer.upstream,
                'buffer_occupancy':self.conveyor_buffer.occupancy,
                **details}
            self.online_plans.append(record)
            self.record_event('ONLINE_PICK_DECISION',**record)
            LOG.info('ONLINE_PICK selected %s from slot %d | visible=%s planner=%s depth=%d',
                record['selected_sku'],slot,visible,details['planner'],details['completed_depth'])
            if self.scene.conveyor_enabled:
                self.scene.shuttle_to_pick(selected,slot,self.step)
                released=self.conveyor_buffer.remove(selected)
                if released!=slot:
                    raise RuntimeError('Physical buffer slot changed during carton transfer')
                self._advance_conveyor()
                yield selected
            else:
                yield selected
                self.conveyor_buffer.remove(selected)
                self._advance_conveyor()

    def run(self):
        scan=self.c['robot']['scan_position'];tool=self.c['robot']['tool_length']
        self.wait(.4)
        for index in self._online_carton_order():
            item=self.scene.inventory[index]
            self.index=index
            self.enter('FEED',mechanism='finite_buffer_shuttle',
                       buffer_occupancy=self.conveyor_buffer.occupancy,
                       visible_indices=self.conveyor_buffer.visible)
            self.enter('SCAN');self.move(scan,self.c['robot']['scan_orientation'])
            pose=self.detect()
            self.scene.track(index,pose.dimensions,pose.position,pose.yaw)
            # Ground truth is read ONLY for evaluation, outside the detector/packer.
            truth=self.p.getBasePositionAndOrientation(self.scene.boxes[index])[0]
            self.observations.append({'index':index,'estimated_center':pose.position.tolist(),
                'estimated_dimensions':pose.dimensions.tolist(),'center_error_m':float(np.linalg.norm(pose.position-truth))})
            from PIL import Image
            Image.fromarray(self.camera.rgb[:,:,:3]).save(self.output/f'pick_{index:02d}.png')
            self.enter('PLAN_PLACEMENT')
            if item['mass']>self.c['process']['max_tool_payload_kg']:
                raise RuntimeError('REJECT_OVERWEIGHT: carton exceeds tool payload metadata limit')
            planned_placement=self.online_placement_targets.pop(index,None)
            placement=reconcile_first_placement(
                self.packer,pose.dimensions,item['mass'],item['capacity'],planned_placement)
            self.record_event('PLACEMENT_RECONCILED',carton_index=int(index),
                planner_target=None if planned_placement is None else planned_placement.record(),
                executed_target=placement.record(),
                target_preserved=bool(planned_placement is not None and
                    placement.support==planned_placement.support and
                    placement.layer==planned_placement.layer and
                    np.allclose(placement.center,planned_placement.center,atol=1e-8) and
                    abs(placement.yaw-planned_placement.yaw)<1e-8))
            grasp_attempts=int(self.c['process'].get('grasp_retries',2))
            for grasp_attempt in range(grasp_attempts+1):
                if grasp_attempt:
                    self.enter('GRASP_RESCAN',attempt=grasp_attempt)
                    self.move(scan,self.c['robot']['scan_orientation'])
                    self.wait(.15)
                    pose=self.detect()
                    self.scene.track(index,pose.dimensions,pose.position,pose.yaw)
                    truth=self.p.getBasePositionAndOrientation(self.scene.boxes[index])[0]
                    self.observations.append({'index':index,'attempt':grasp_attempt,
                        'estimated_center':pose.position.tolist(),'estimated_dimensions':pose.dimensions.tolist(),
                        'center_error_m':float(np.linalg.norm(pose.position-truth))})
                    placement=reconcile_first_placement(
                        self.packer,pose.dimensions,item['mass'],item['capacity'],planned_placement)
                pickq=multiply(yaw_quaternion(pose.yaw),DOWN)
                grasp=np.r_[pose.position[:2],pose.top_z+tool+.006]
                self.enter('APPROACH',attempt=grasp_attempt)
                self.move(grasp+[0,0,.13],pickq)
                # Repeatable regression hook: an external nudge after visual
                # localization but before descent. It is off unless requested.
                if self.disturb_box and index==0 and not self.disturbance_injected:
                    body=self.scene.boxes[index];bp,bq=self.p.getBasePositionAndOrientation(body)
                    yaw=self.p.getEulerFromQuaternion(bq)[2]+.28
                    self.p.resetBasePositionAndOrientation(body,[bp[0]+.045,bp[1]-.035,bp[2]],
                        self.p.getQuaternionFromEuler([0,0,yaw]))
                    self.p.resetBaseVelocity(body,[0,0,0],[0,0,0])
                    self.disturbance_injected=True
                    self.enter('INJECTED_DISTURBANCE',offset_m=[.045,-.035],yaw_delta_rad=.28)
                self.move(grasp,pickq,True)
                self.enter('GRASP',attempt=grasp_attempt)
                try:
                    self.gripper.close(index,self.time,pose)
                    self.wait(.10)
                    break
                except RuntimeError as exc:
                    if 'Vacuum seal geometry invalid' not in str(exc):
                        raise
                    if grasp_attempt>=grasp_attempts:
                        raise RuntimeError(f'GRASP_RETRIES_EXHAUSTED after {grasp_attempt+1} attempts: {exc}') from exc
                    self.enter('GRASP_RETRY',attempt=grasp_attempt+1,reason=str(exc))
                    self.move(grasp+[0,0,.15],pickq,True)
            self.enter('LIFT')
            lift_z=max(1.10,placement.center[2]+placement.dimensions[2]/2+tool+.18)
            self.move(np.r_[grasp[:2],lift_z],pickq,True)
            # The chosen placement yaw is relative to the measured carton axes.
            placeq=multiply(yaw_quaternion(placement.yaw),DOWN)
            target=np.array(placement.center)+[0,0,placement.dimensions[2]/2+tool+.012]
            self.enter('TRANSFER');self.robot.transport(np.r_[target[:2],lift_z],placeq,self.step)
            self.enter('PLACE');self.move(target,placeq,True)
            self.enter('RELEASE');self.gripper.open()
            self.scene.track(index,placement.dimensions,placement.center,0.0)
            self.move(target+[0,0,.12],placeq,True)
            self.wait(self.c['process']['settle_seconds'])
            self.enter('VERIFY');self.scene.verify_placement(index,placement)
            self.packer.commit(placement)
            self.scene.record_pallet_placement(placement.index,index)
            self.enter('RETURN');self.move(scan,self.c['robot']['scan_orientation'])
        self.wait(.5)
        for placement in self.packer.placements:
            carton_index=self.scene.pallet_carton_by_placement[placement.index]
            self.scene.verify_placement(carton_index,placement)
        self.enter('DONE')

    def report(self):
        return {'state':self.state,'seed':self.c['simulation']['seed'],'simulated_seconds':self.time,
            'requested':len(self.scene.inventory),'committed':len(self.packer.placements),
            'verified_placements':list(self.scene.verified.values()),
            'placements':[p.record() for p in self.packer.placements],
            'perception_checks':self.observations,'planning':self.robot.planner.plans,
            'packing_volume_utilization':self.packer.utilization(),
            'peak_vacuum_force_n':self.gripper.peak_force,'peak_vacuum_torque_nm':self.gripper.peak_torque,
            'remaining_constraints':self.p.getNumConstraints() if self.p.isConnected() else None,
            'collision_world_source':'static CAD + conservative infeed envelope + perceived carton poses/dimensions + commanded placements; no camera occupancy reconstruction',
            'conveyor_model':{'enabled':self.scene.conveyor_enabled,
                'feed_type':'finite selectable accumulation buffer with simulated random arrivals, upstream back-pressure, and overhead shuttle to a single depth-camera pick station',
                'arrival_interval_s':float(self.c['conveyor'].get('arrival_interval_s',2.5)),
                'arrival_jitter_fraction':float(self.c['conveyor'].get('arrival_jitter_fraction',.25)),
                'planner':self.planner_mode,
                'buffer':self.conveyor_buffer.record(),
                'online_pick_decisions':self.online_plans,
                'infeed_dimensioner_measurements':self.dimensioner_measurements,
                'size_source':'simulated infeed dimensioner measurements; carton mass/capacity from SKU metadata',
                'conveyor_actuation':'kinematic transport on a static collision deck; drive/roller dynamics are not simulated'},
            'robot_model':'nominal UR10e; no hardware calibration or drive validation',
            'peak_joint_speed_rad_s':self.peak_joint_speed.tolist(),
            'predict_latency_ms':getattr(self.perception.camera,'latencies',[]),
            'events':self.events}
