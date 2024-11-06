import plotly.graph_objects as go

from utils import log, progress


def plot(figure, kickaxis = True):
    figure.update_layout(
        margin = dict(t=0, b=0, l=0, r=0),
        paper_bgcolor = "#292D3E",
        plot_bgcolor = "#000000",
        font_color = "white",
        
    )
    if kickaxis :
        figure.update_scenes(xaxis_visible = False, yaxis_visible = False, zaxis_visible = False)
    log('Sending figure data in browser window ... ')
    figure.show()
    
    
def draw_3d_simulation(sim_output, r_decim = 10, trace_len = 50, fps = 60):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=motion['x'], y=motion['y'], z=motion['z'],
                name=body, 
                mode="lines",
            ) for body, motion in sim_output.data.items()
        ],
        frames=render_frames(sim_output, r_decim, trace_len),
        
        layout=go.Layout(
            scene = dict(aspectratio = dict(x = 1, y = 1, z = 1)),
            annotations = [dict(
                x=0, y=0.985, xref="paper", yref="paper", showarrow=False,
                text=f'Frames: ({0}/{sim_output.nsteps/r_decim}) | Time: ({0}/{sim_output.total_time}) days',
                font=dict(size=15)
            )],
            updatemenus= [{
                'buttons': [
                    {
                        'args': [
                            None, 
                            {'frame': {'duration': 1000//fps, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 500, 'easing': 'quadratic-in-out'}},
                        ], 
                        'label': 'Play', 
                        'method': 'animate'
                    },
                    {
                        'args': [
                            [None], 
                            {'frame': {'duration': 1000//fps, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 500}},
                        ],
                        'label': 'Pause', 
                        'method': 'animate',
                    }
                ],
                'direction': 'left', 'pad': {'l': 10, 'r': 10, 't': 10}, 'showactive': False, 'type': 'buttons'
            }]
        )
    )
    return fig

import numpy as np 

def render_frames(sim_output, r_decim, trace_len):
    frames = []
    render_steps = len(sim_output.data[next(iter(sim_output.data))]['x'])//(r_decim)
    with progress:
        log('Rendering Plotly frames')
        frames_rendering = progress.add_task('Plotly frames rendering ... ', total = render_steps)
        l_index = 0
        for f in range(render_steps):
            h_index = r_decim * f
            if f > trace_len:
                l_index = r_decim * (f-trace_len)
            rot = .5
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter3d(
                            x = motion['x'][l_index:h_index], y = motion['y'][l_index:h_index], z = motion['z'][l_index:h_index],
                            mode = "lines", 
                            line = dict(color=None, width=2),
                        ) for motion in sim_output.data.values()
                    ],
                    layout = go.Layout(
                        # scene_camera = dict(
                        #     eye=dict(x=1 * np.cos(rot*f*2*np.pi/render_steps), y=1 * np.sin(rot*f*2*np.pi/render_steps), z=1)
                        # ),
                        annotations = [dict(
                            x=0, y=0.985, xref="paper", yref="paper", showarrow=False,
                            text=f'Frames: ({f}/{sim_output.nsteps/r_decim}) | Time: ({f*sim_output.total_time/sim_output.nsteps}/{sim_output.total_time}) days',
                            font=dict(size=15)
                        )]
                    )
                )
            )
            progress.advance(frames_rendering, 1)
    progress.remove_task(frames_rendering)
    return frames




