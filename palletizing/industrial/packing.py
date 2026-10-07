"""Online axis-aligned packing with explicit support, CoM, and load limits.

This is a deterministic feasible-placement heuristic, not an optimal bin packer.
Dimensions come from perception; mass and crush limits are SKU metadata.
"""
from dataclasses import dataclass, asdict
import numpy as np


class PalletFull(RuntimeError):
    pass


@dataclass
class Placement:
    index: int
    center: list
    dimensions: list
    yaw: float
    support: int | None
    layer: int
    mass: float
    capacity: float
    load_above: float = 0.0
    support_fraction: float = 1.0

    def record(self):
        return asdict(self)


class SupportPacker:
    def __init__(self, center, size, base_z, max_height=0.85, gap=0.012,
                 support_margin=0.008, max_layers=3, max_payload=80.0,
                 min_support_fraction=1.0, max_overhang=0.0):
        self.center = np.asarray(center, float)
        self.size = np.asarray(size, float)
        self.base_z, self.max_height = float(base_z), float(max_height)
        self.gap, self.margin = float(gap), float(support_margin)
        self.max_layers, self.max_payload = int(max_layers), float(max_payload)
        self.min_support_fraction = float(min_support_fraction)
        self.max_overhang = float(max_overhang)
        if not 0 < self.min_support_fraction <= 1 or not np.isfinite(max_overhang) or max_overhang < 0:
            raise ValueError("Invalid support fraction or overhang")
        self.placements = []
        if (self.center.shape != (2,) or self.size.shape != (2,)
            or not np.isfinite(np.r_[self.center, self.size, base_z, max_height, gap,
                                      support_margin, max_payload]).all()
            or np.any(self.size <= 0) or min(max_height, gap, support_margin) < 0
            or max_layers < 1 or max_payload <= 0):
            raise ValueError('Invalid packing bounds')

    def _ancestors(self, support):
        while support is not None:
            p = self.placements[support]
            yield p
            support = p.support

    def candidates(self, dimensions, mass, capacity):
        d = np.asarray(dimensions, float)
        if d.shape != (3,) or not np.isfinite(d).all() or np.any(d <= 0):
            raise ValueError('Dimensions must be three finite positive values')
        if not np.isfinite([mass, capacity]).all() or mass <= 0 or capacity < 0:
            raise ValueError('Invalid carton mass or crush capacity')
        if sum(p.mass for p in self.placements) + mass > self.max_payload:
            return []
        platforms = [(None, self.center-self.size/2, self.center+self.size/2,
                      self.base_z, 0)]
        for p in self.placements:
            h = np.array(p.dimensions[:2])/2 + self.max_overhang
            if self.min_support_fraction == 1.0:
                h -= self.margin
            platforms.append((p.index, np.array(p.center[:2])-h,
                              np.array(p.center[:2])+h,
                              p.center[2]+p.dimensions[2]/2, p.layer+1))
        out = []
        for yaw, dims in [(0.0, d), (np.pi/2, d[[1, 0, 2]])]:
            for support, low, high, z, layer in platforms:
                if layer >= self.max_layers or z+dims[2] > self.base_z+self.max_height:
                    continue
                if any(p.load_above+mass > p.capacity for p in self._ancestors(support)):
                    continue
                lower, upper = low+dims[:2]/2, high-dims[:2]/2
                if np.any(lower > upper+1e-8):
                    continue
                xs, ys = {float(lower[0]), float(upper[0])}, {float(lower[1]), float(upper[1])}
                for p in self.placements:
                    for sign in [-1, 1]:
                        xs.add(float(p.center[0]+sign*(p.dimensions[0]/2+dims[0]/2+self.gap)))
                        ys.add(float(p.center[1]+sign*(p.dimensions[1]/2+dims[1]/2+self.gap)))
                for x in sorted(xs):
                    for y in sorted(ys):
                        xy = np.array([x, y])
                        if np.any(xy < lower-1e-8) or np.any(xy > upper+1e-8):
                            continue
                        if np.any(xy-dims[:2]/2 < self.center-self.size/2-1e-8) or np.any(xy+dims[:2]/2 > self.center+self.size/2+1e-8):
                            continue
                        fraction=1.0
                        if support is not None:
                            parent=self.placements[support]
                            plow=np.array(parent.center[:2])-np.array(parent.dimensions[:2])/2
                            phigh=np.array(parent.center[:2])+np.array(parent.dimensions[:2])/2
                            contact_low=np.maximum(xy-dims[:2]/2,plow)
                            contact_high=np.minimum(xy+dims[:2]/2,phigh)
                            fraction=float(np.prod(np.maximum(0,contact_high-contact_low))/np.prod(dims[:2]))
                            if fraction < self.min_support_fraction-1e-8 or np.any(xy < contact_low+self.margin) or np.any(xy > contact_high-self.margin):
                                continue
                            # Check aggregate uniform-density stack CoM at every ancestral interface.
                            stable=True
                            for ancestor in self._ancestors(support):
                                subtree=[other for other in self.placements if other.index==ancestor.index or any(a.index==ancestor.index for a in self._ancestors(other.support))]
                                com=(sum((other.mass*np.array(other.center[:2]) for other in subtree),np.zeros(2))+mass*xy)/(sum(other.mass for other in subtree)+mass)
                                alow=np.array(ancestor.center[:2])-np.array(ancestor.dimensions[:2])/2
                                ahigh=np.array(ancestor.center[:2])+np.array(ancestor.dimensions[:2])/2
                                if ancestor.support is not None:
                                    lower_box=self.placements[ancestor.support]
                                    alow=np.maximum(alow,np.array(lower_box.center[:2])-np.array(lower_box.dimensions[:2])/2)
                                    ahigh=np.minimum(ahigh,np.array(lower_box.center[:2])+np.array(lower_box.dimensions[:2])/2)
                                if np.any(com < alow+self.margin) or np.any(com > ahigh-self.margin):
                                    stable=False;break
                            if not stable:continue
                        center = np.r_[xy, z+dims[2]/2]
                        valid = True
                        for p in self.placements:
                            delta = np.abs(center-np.array(p.center))
                            extent = (dims+np.array(p.dimensions))/2
                            # Touching a support face is intentional. Side faces get clearance.
                            if delta[2] < extent[2]-1e-6 and np.all(delta[:2] < extent[:2]+self.gap-1e-6):
                                valid = False
                                break
                        if valid:
                            out.append(Placement(len(self.placements), center.tolist(),
                                                 dims.tolist(), float(yaw), support, layer,
                                                 float(mass), float(capacity), support_fraction=fraction))
        # Lowest stable layer first; nearer robot next. This is a heuristic objective.
        out.sort(key=lambda p: (round(p.center[2]-p.dimensions[2]/2, 5),
                                np.linalg.norm(p.center[:2]), p.yaw, p.center[0]))
        return out

    def propose(self, dimensions, mass=1.0, capacity=12.0):
        candidates = self.candidates(dimensions, mass, capacity)
        if not candidates:
            raise PalletFull('No support-, stability-, and load-compliant placement within pallet bounds')
        return candidates[0]

    def commit(self, placement):
        if placement.index != len(self.placements):
            raise ValueError('Stale or duplicate placement commit')
        self.placements.append(placement)
        for parent in self._ancestors(placement.support):
            parent.load_above += placement.mass

    def utilization(self):
        if not self.placements:
            return 0.0
        height = max(p.center[2]+p.dimensions[2]/2 for p in self.placements)-self.base_z
        return float(sum(np.prod(p.dimensions) for p in self.placements)/(np.prod(self.size)*height))


@dataclass
class BatchPickPlan:
    """Feasible pick sequence selected from a known carton manifest."""
    order: list
    placements: list
    strategy: str
    layers: int
    max_height: float
    utilization: float
    minimum_support_fraction: float
    search_evaluations: int = 0

    def record(self, items):
        return {
            'strategy': self.strategy,
            'order': [
                {'manifest_index': int(i), 'sku': items[i].get('sku', str(i)),
                 'placement': placement.record()}
                for i, placement in zip(self.order, self.placements)
            ],
            'planned_count': len(self.order),
            'layer_count': self.layers,
            'max_height_m': self.max_height,
            'volume_utilization': self.utilization,
            'minimum_support_fraction': self.minimum_support_fraction,
            'search_evaluations': self.search_evaluations,
        }


def optimize_batch_order(template, items, seed=0, random_trials=4,
                         local_search_iterations=32):
    """Choose a feasible carton order using bounded seeded iterated local search.

    Each permutation is greedily packed with the same support, stability, crush,
    footprint, and height checks used by the live cell. Starts include common
    industrial sorting rules and seeded permutations; swap/relocate neighbors
    then improve the best feasible start. This searches carton order only: it is
    not layer column generation, does not optimize robot travel, and is not an
    optimal mixed-integer or online lookahead solver.
    """
    items=list(items)
    if not items:
        raise ValueError('Cannot plan an empty carton manifest')
    dims=[]
    for i,item in enumerate(items):
        d=np.asarray(item.get('dimensions'),dtype=float)
        mass=float(item.get('mass',1.0));capacity=float(item.get('capacity',12.0))
        if d.shape!=(3,) or not np.isfinite(d).all() or np.any(d<=0):
            raise ValueError(f'Invalid dimensions for manifest item {i}')
        if not np.isfinite([mass,capacity]).all() or mass<=0 or capacity<0:
            raise ValueError(f'Invalid mass/capacity for manifest item {i}')
        dims.append(d)

    strategies=[]
    indices=list(range(len(items)))
    strategies.append(('manifest',indices))
    strategies.append(('footprint_desc',sorted(indices,key=lambda i:(-dims[i][0]*dims[i][1],-np.max(dims[i]),i))))
    strategies.append(('volume_desc',sorted(indices,key=lambda i:(-np.prod(dims[i]),i))))
    strategies.append(('height_desc',sorted(indices,key=lambda i:(-dims[i][2],-dims[i][0]*dims[i][1],i))))
    strategies.append(('longest_edge_desc',sorted(indices,key=lambda i:(-np.max(dims[i]),-np.prod(dims[i]),i))))
    strategies.append(('small_first',sorted(indices,key=lambda i:(dims[i][0]*dims[i][1],np.prod(dims[i]),i))))
    rng=np.random.default_rng(seed)
    for trial in range(max(0,int(random_trials))):
        order=indices.copy();rng.shuffle(order)
        strategies.append((f'seeded_shuffle_{trial}',order))

    def fresh_packer():
        return SupportPacker(template.center.copy(),template.size.copy(),template.base_z,
            template.max_height,template.gap,template.margin,template.max_layers,
            template.max_payload,template.min_support_fraction,template.max_overhang)

    best_partial=0;evaluations=0;feasible=[]

    def evaluate(name,order):
        nonlocal best_partial,evaluations
        evaluations+=1
        packer=fresh_packer();chosen=[];ok=True
        for index in order:
            item=items[index]
            try:
                placement=packer.propose(dims[index],float(item.get('mass',1.0)),
                                         float(item.get('capacity',12.0)))
            except PalletFull:
                ok=False;best_partial=max(best_partial,len(chosen));break
            packer.commit(placement);chosen.append(placement)
        if not ok:
            return None
        height=max(p.center[2]+p.dimensions[2]/2 for p in packer.placements)-packer.base_z
        layers=max(p.layer for p in packer.placements)+1
        utilization=packer.utilization()
        support_fractions=[p.support_fraction for p in packer.placements]
        minimum_support=min(support_fractions)
        mean_support=float(np.mean(support_fractions))
        score=(round(height,6),layers,-round(minimum_support,6),
               -round(mean_support,6),-round(utilization,6),name)
        return score,BatchPickPlan(order.copy(),chosen,name,layers,
                         float(height),utilization,float(minimum_support))

    for name,order in strategies:
        result=evaluate(name,order)
        if result is not None:
            feasible.append(result)
    if not feasible:
        raise PalletFull(f'No tested pick order places the full manifest ({best_partial}/{len(items)} cartons); increase pallet capacity or reduce the batch')
    feasible.sort(key=lambda x:x[0])
    best_score,best_plan=feasible[0]
    current_order=best_plan.order.copy();current_score=best_score
    current_plan=best_plan

    # A small deterministic simulated-annealing walk escapes the initial sort
    # orders while retaining the best fully feasible plan seen so far.
    search_rng=np.random.default_rng(seed+0x5EED)
    budget=max(0,int(local_search_iterations))
    for iteration in range(budget):
        neighbor=current_order.copy()
        if len(neighbor)<2:
            break
        a,b=search_rng.choice(len(neighbor),size=2,replace=False)
        if search_rng.random()<0.5:
            neighbor[a],neighbor[b]=neighbor[b],neighbor[a]
        else:
            value=neighbor.pop(int(a));neighbor.insert(int(b),value)
        result=evaluate(f'iterated_local_search_{iteration}',neighbor)
        if result is None:
            continue
        score,plan=result
        if score<best_score:
            best_score,best_plan=score,plan
        def energy(value):
            return value[0]+0.003*value[1]+0.001*value[2]+0.0005*value[3]
        temperature=0.002*(1.0-0.9*(iteration+1)/max(1,budget))
        delta=energy(score)-energy(current_score)
        if delta<0 or search_rng.random()<np.exp(-delta/temperature):
            current_order,current_score,current_plan=neighbor,score,plan
        if (iteration+1)%12==0:
            current_order=best_plan.order.copy();current_score=best_score;current_plan=best_plan
    best_plan.search_evaluations=evaluations
    return best_plan
