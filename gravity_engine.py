from __future__ import annotations
import numpy as np
from numpy import pi
import pandas as pd
from rich.console import Console
from rich.progress import (Progress, ProgressBar, SpinnerColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
import humanize
from utils import *
from rendering import *
header()

class vect:
    def __init__(self,
            x: float | None = None,
            y: float | None = None,
            z: float | None = None,
            r: float | None = None,
            th: float | None = None,
            ph: float | None = None,
            pcoords: bool = False,
        ):
        
        if (x and y and z) != None:
            self.x = x
            self.y = y
            self.z = z
            # self._cart2sph()
            
        elif (r and th and ph) != None:
            self.r = r
            self.th = th
            self.ph = ph
            self._sph2cart()
            
        else:
            log('[red]Conflicts in vect object declaration ...')
            raise ValueError('Initialize vect() object using either cartesian (x, y, z) or spherical (r, th, ph) coordinates')
        
        if pcoords: 
            log(self.coords(), t = .5)
        
    def _cart2sph(self):
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.th = np.arccos(self.z / self.r) if self.r != 0 else float(0)
        self.ph = np.arctan2(self.y, self.x)
        
    def _sph2cart(self):
        self.x = self.r * np.cos(self.th) * np.sin(self.ph)
        self.y = self.r * np.sin(self.th) * np.sin(self.ph)
        self.z = self.r * np.cos(self.ph)
    
    def coords(self):
        table = Table(box = ROUNDED)
        for col in ['Cartesian (x, y, z)', 'Spherical (r, θ, φ)']: table.add_column(col)
        self._cart2sph()
        table.add_row(f'({self.x:.2e}, {self.y:.2e}, {self.z:.2e})', f'({self.r:.2e}, {self.th:.2e}, {self.ph:.2e})')
        return table
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> vect:
        return self / self.magnitude()

    def __str__(self): # printable format
        return f'({self.x:.2e}, {self.y:.2e}, {self.z:.2e})'

    def __eq__(self, v2: vect) -> bool: # '==' operator
        if isinstance(v2, vect): return self.x == v2.x and self.y == v2.y and self.z == v2.z
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')

    def __ne__(self, v2: vect) -> vect: # '!=' operator
        if isinstance(v2, vect): return self.x != v2.x or self.y != v2.y or self.z != v2.z
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')

    def __add__(self, v2: vect) -> vect: # '+' operator
        if isinstance(v2, vect): return vect(self.x + v2.x, self.y + v2.y, self.z + v2.z)
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')
        
    def __iadd__(self, v2: vect) -> vect: # '+=' operator
        if isinstance(v2, vect): return vect(self.x + v2.x, self.y + v2.y, self.z + v2.z)
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')

    def __sub__(self, v2: vect) -> vect: # '-' operator
        if isinstance(v2, vect): return vect(self.x - v2.x, self.y - v2.y, self.z - v2.z)
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')
    
    def __isub__(self, v2: vect) -> vect: # '-=' operator
        if isinstance(v2, vect): return vect(self.x - v2.x, self.y - v2.y, self.z - v2.z)
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')

    def __mul__(self, scale: float | int = 10) -> vect: # '*' operator
        if isinstance(scale, float): return vect(self.x * scale, self.y * scale, self.z * scale)
        else: raise TypeError(f'{scale} ({type(scale)}) must be of type float or int')

    def __imul__(self, scale: float | int = 10) -> vect: # '*=' operator
        if isinstance(scale, float): return vect(self.x * scale, self.y * scale, self.z * scale)
        else: raise TypeError(f'{scale} ({type(scale)}) must be of type float or int')

    def __truediv__(self, scale: float | int = 10) -> vect: # '/' operator
        if isinstance(scale, float): return vect(self.x / scale, self.y / scale, self.z / scale)
        else: raise TypeError(f'{scale} ({type(scale)}) must be of type float or int')

    def __itruediv__(self, scale: float | int = 10) -> vect: # '/=' operator
        if isinstance(scale, float): return vect(self.x / scale, self.y / scale, self.z / scale)
        else: raise TypeError(f'{scale} ({type(scale)}) must be of type float or int')

    def __xor__(self, v2: vect) -> vect: # '^' operator
        if isinstance(v2, vect): return vect(self.y * v2.z - self.z * v2.y, self.z * v2.x - self.x * v2.z, self.x * v2.y - self.y * v2.x)
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')

    def __ixor__(self, v2: vect) -> vect: # '^=' operator
        if isinstance(v2, vect): return vect(self.y * v2.z - self.z * v2.y, self.z * v2.x - self.x * v2.z, self.x * v2.y - self.y * v2.x)
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')

    def __matmul__(self, v2: vect) -> float: # '@' operator
        if isinstance(v2, vect): return self.x * v2.x + self.y * v2.y + self.z * v2.z
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')

    def __imatmul__(self, v2: vect) -> float: # '@=' operator
        if isinstance(v2, vect): return self.x * v2.x + self.y * v2.y + self.z * v2.z
        else: raise TypeError(f'{v2} ({type(v2)}) must be of type vect')
        
        
        
class body:
    def __init__(self, 
            location: vect = vect(0, 0, 0),      
            velocity: vect = vect(0, 0, 0), 
            mass: float = 0, 
            name: str | None = None,
            motionFlag: bool = True
        ):
        self.location = location
        self.velocity = velocity
        self.mass = mass
        self.name = name
        self.motionFlag = motionFlag

    def dist(self, target: body):
        return self.location - target.location
        
    def checkPos(self, comp) -> bool:
        # self.motionFlag = np.max(np.abs([np.sqrt(self.location.x**2 + self.location.y**2), self.location.z*5])) < comp
        if self.motionFlag: 
            self.motionFlag = self.location.magnitude() < comp
            if not self.motionFlag: console.log(f'body {self} max loc magnitude() ({self.location.magnitude():.2e} > {comp:.2e})')
        else: 
            pass
        
    def getEp(self, bodies) -> float:
        self.Ep = 0
        for ext_body in [b for b in bodies if b!=self]:
            self.Ep += -6.67e-11 * self.mass * ext_body.mass / self.dist(ext_body).magnitude()
        return self.Ep

    def getEc(self) -> float:
        self.Ec = self.mass * self.velocity.magnitude()**2       
        return self.Ec

    def __str__(self):
        return f'{self.name} - loc {self.location}/vel {self.velocity}'



class BodyGenerator():
    def __init__(self,
            vz: int = 1000,
        ):
        self.vz = vz
        self.star = body(velocity = vect(0, 0, self.vz), mass = 1e30, name = 'Star')
        
    def clusters_array(self, 
            lp: tuple[float] = [1e11, 10],
            vp: tuple[float] = [1e3, 10],
            mp: tuple[float] = [1e26, 30],
            arr_disp: float = .8, 
            n_cluster: int = 5, 
            n_array: int = 3, 
        ):
        bodies_cluster = []
        for c in range(n_cluster):
            theta = c*2*pi/n_cluster
            bodies_cluster += self.body_cluster(
                n_bodies = n_array,
                cloc = vect(r = lp[0], th = theta, ph = 0),
                cvel = vect(r = 0, th = self.circular_vel(self.star.mass, lp[0]), ph = self.vz),
                dloc = vect(r = lp[1], th = theta, ph = 0),
                dvel = vect(r = 0, th = vp[1], ph = 0),
                mass = (1e24, 1e23)
            )
        bodies_cluster.append(self.star)
        return bodies_cluster

    def body_cluster(self, 
            n_bodies, 
            cloc: vect,
            cvel: vect,
            dloc: vect,
            dvel: vect,
            mass: tuple[float],
            cluster_index = 0,
        ):
        bodies = [body(name = f'Body n°{1+index}.{cluster_index}') for index in range(n_bodies)]
        for cluster_body in bodies:
            cluster_body.velocity = cvel + dvel * np.random.normal(0, 1)
            cluster_body.location = cloc + dloc * np.random.normal(0, 1)
            cluster_body.mass = np.random.normal(loc = mass[0], scale = mass[1])
        return bodies

    def random_bodies(self, 
            n_bodies: int = 10,
            lp: tuple[float] = [1e11, 10],
            vp: tuple[float] = [1e3, 10],
            mp: tuple[float] = [1e26, 30],
            rseed: int | None = None,
        ):
        if rseed:
            np.random.seed(rseed)
            console.print(f'rseed: {rseed}')
            
        random_bodies = [body(location = vect(0, 0, 0), mass = 0, velocity = vect(0, 0, 0), name = 'Body n°%i'%(1+index)) for index in range(n_bodies)]
        
        for cluster_body in random_bodies:
            loc = vect(r = np.random.normal(lp[0], lp[0]/lp[1]), th = np.random.uniform(0, pi), ph = np.random.uniform(0, 2*pi))
            vel = vect(r = np.random.normal(vp[0], vp[0]/vp[1]), th = np.random.uniform(0, pi), ph = np.random.uniform(0, 2*pi))
            cluster_body.location = loc
            cluster_body.velocity = vel ^ (loc.normalize())
            cluster_body.mass = np.random.normal(mp[0], mp[0]/mp[1])
        return random_bodies

    def circular_vel(self, M, r, gain = 1):
        return gain*np.sqrt(6.67e-11*M/r)
    
class Motions:
    def __init__(self, nsteps: int, total_time: int):
        self.data = {}
        self.nsteps = nsteps
        self.total_time = total_time

class SimEngine():
    def __init__(self, 
            bodies: list = [], 
            nsteps: int = 1000, 
            dt: float = 1, 
            s_decim: int = 2, 
            thrsh: float = 10e12, 
            g_value: float = 6.6743e-11, 
        ):
        self.bodies = bodies
        self.nsteps = nsteps
        self.s_decim = s_decim
        self.dt = dt * 24 * 3600
        self.thrsh = thrsh
        self.G = g_value
        self.output = Motions(self.nsteps/self.s_decim, int(self.nsteps*dt))
        
    def btable_rowc(self, value, mean, thrsh: float = .05) -> str:
        if value == 0:
            return 'reset'
        ratio = value/mean
        color = 'orange1'
        if ratio > 1+thrsh: color = 'green'
        elif ratio < 1-thrsh: color = 'red'
        return color        
    
    def input_table(self):
        table = Table(title = 'Simulated bodies', title_justify = 'left', box = ROUNDED)
        for col in ['Name', 'Mass (kg)', 'Location (m)', 'Velocity (m/s)', 'Ecᵢ/Epᵢ (TJ)', 'Vmagᵢ (m/s)']: table.add_column(col)
        avg_m = np.mean([b.mass for b in self.bodies])
        avg_s = np.mean([b.velocity.magnitude() for b in self.bodies])
        avg_ec = np.mean([b.getEc() for b in self.bodies])
        avg_ep = np.mean([b.getEp(self.bodies) for b in self.bodies])
        avg_er = -avg_ec/avg_ep
        avg_l = [np.mean([b.location.x for b in self.bodies]), np.mean([b.location.y for b in self.bodies]), np.mean([b.location.z for b in self.bodies])]
        avg_v = [np.mean([b.velocity.x for b in self.bodies]), np.mean([b.velocity.y for b in self.bodies]), np.mean([b.velocity.z for b in self.bodies])]
        for body in self.bodies:
            mc = self.btable_rowc(body.mass, avg_m)
            sc = self.btable_rowc(body.velocity.magnitude(), avg_s)
            ecc = self.btable_rowc(body.getEc(), avg_ec)
            epc = self.btable_rowc(body.getEp(self.bodies), avg_ep)
            erc = self.btable_rowc(body.getEc(), -body.getEp(self.bodies), avg_er)
            lc = [self.btable_rowc(body.location.x, avg_l[0]), self.btable_rowc(body.location.y, avg_l[1]), self.btable_rowc(body.location.z, avg_l[2])]
            vc = [self.btable_rowc(body.velocity.x, avg_v[0]), self.btable_rowc(body.velocity.x, avg_v[1]), self.btable_rowc(body.velocity.z, avg_v[2])]
            table.add_row(
                f'[bold]{body.name}',
                f'[{mc}]%.2e'%(body.mass),
                f'([{lc[0]}]%.2e[/], [{lc[1]}]%.2e[/], [{lc[2]}]%.2e[/])'%(body.location.x, body.location.y, body.location.z),
                f'([{vc[0]}]%.2e[/], [{vc[1]}]%.2e[/], [{vc[2]}]%.2e[/])'%(body.velocity.x, body.velocity.y, body.velocity.z),
                f'[{ecc}]%.2e[/]/[{epc}]%.2e[/] ([{erc}]%.2f[/])'%(body.getEc()/1e12, body.getEp(self.bodies)/1e12, -body.getEc()/body.getEp(self.bodies)),
                f'[{sc}]%.2e'%(body.velocity.magnitude()),
            )
        return table

    def simulation_table(self):
        table = Table(title = 'Simulation parameters', title_justify = 'left', box = ROUNDED)
        for col in ['StepsCnt', 'Timestep', 'SimTime', 'Decimation', 'BodyQty']: table.add_column(col)
        table.add_row(f'{self.nsteps} steps', f'{self.dt} days', f'{self.nsteps*self.dt} days', f'{self.s_decim}', f'{len(self.bodies)}')
        return table
    
    def compute_gravity(self, target: body):
        acc = vect(0,0,0)
        for ext_body in [b for b in self.bodies if b != target]:
            r = (target.dist(ext_body)).magnitude()
            g_value = ext_body.mass / r**3
            acc += (ext_body.location - target.location) * g_value
        return acc * self.G

    def update_system(self):
        for target in self.bodies:
            target.checkPos(self.thrsh)
            if target.motionFlag: 
                target.velocity += self.compute_gravity(target) * self.dt
        for target in self.bodies: 
            if target.motionFlag: 
                target.location += target.velocity * self.dt

    def run_simulation(self):
        
        log('Resolving body motions for the following array ... ')
        log(self.simulation_table())
        log(self.input_table())
        
        self.output.data = {}
        for cluster_body in self.bodies: 
            self.output.data[cluster_body.name] = {'x':[], 'y':[], 'z':[]}
        
        with progress:
            tstart = time.time()
            steps_computing = progress.add_task('Gravity steps computing ... ', total = self.nsteps//self.s_decim)
            for i in range(self.nsteps):
                self.update_system()
                if i % self.s_decim == 0:
                    progress.advance(steps_computing, 1)
                    for index, motion in enumerate(self.output.data.values()):
                        motion['x'].append(self.bodies[index].location.x)
                        motion['y'].append(self.bodies[index].location.y)           
                        motion['z'].append(self.bodies[index].location.z)
            log(f'Elapsed time during sim: {(time.time()-tstart)*1e3} ms')
        progress.remove_task(steps_computing)
        log(f'Motion array generated ({humanize.naturalsize(8*3*self.nsteps*len(self.bodies))} data computed in simulation)')
        
        return self.output
    
    def motion_df(self, name: str, verbose: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(self.output.data[name])
        if verbose:
            with pd.option_context('display.float_format', '{:0.2e}'.format):
                console.print(df)
        return df
    
    def motions_df(self) -> pd.DataFrame:
        dfs = []
        for body_name in self.output.data.keys():
            df = self.motion_df(body_name, verbose = False)
            df.columns = pd.MultiIndex.from_product([[body_name], df.columns])
            dfs.append(df)
        mdf = pd.concat(dfs, axis=1)
        with pd.option_context('display.float_format', '{:0.2e}'.format):
            console.print(mdf)
        return mdf
    
        

if __name__ == '__main__':
    test_bodies = [
        body(location = vect(0, 0, 0), mass = 2e30, velocity = vect(0, 0, 1000), name = 'Sun'), 
        body(location = vect(0, 150e9, 0), mass = 6e24, velocity = vect(29.8e3, 0, 1000), name = 'Earth'), 
    ]
    
    custom_bodies = [
        body(location = vect(1e10, 0, 0), mass = 3e26, velocity = vect(0, 1000, 0), name = 'Body n°1'), 
        body(location = vect(-1e10, 0, 0), mass = 3e26, velocity = vect(0, -1000, 0), name = 'Body n°2'), 
        body(location = vect(0, 0, 1e10), mass = 3e26, velocity = vect(-1000, 0, 0), name = 'Body n°3')
    ]
    
    bgen = BodyGenerator()
    
    engine = SimEngine(
        nsteps = 3 * 1000, 
        dt = .5, 
        s_decim = 2,
        thrsh = 3e11,
    )
    engine.bodies = custom_bodies
    # engine.bodies = test_bodies
    
    # engine.thrsh = 5e12 
    
    # for i in range(1):
    #     engine.bodies = bgen.random_bodies(
    #         n_bodies = 9, 
    #         lp = [10e9, 10],
    #         vp = [10e2, 10],
    #         mp = [30e24, 30],
    #         rseed = np.random.randint(0, 1e9)
    #     )
    #     engine.thrsh = 3e10
        
    motions = engine.run_simulation()
    fig = draw_3d_simulation(motions, r_decim = 5, trace_len = 50, fps = 60)
    plot(fig)
    
    gdf = engine.motions_df()
    
    
    
    # from rich.prompt import Prompt, IntPrompt, FloatPrompt
    
    # def interactive_run():
    #     log('[yellow]Engine interactive command loop started ...')
    #     try:
    #         while True:
    #            cmd = Prompt(
    #                prompt = 'Enter engine command',
    #                choices = ['set']
    #            ) 
    #     except KeyboardInterrupt:
    #         log('[yellow]Engine interactive command loop shut down after KerboardInterrupt')