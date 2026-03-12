import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '16'
import torch
import gradio as gr
from freesplatter.webui.runner import FreeSplatterRunner
from freesplatter.webui.tab_views_to_3d import create_interface_views_to_3d


torch.set_grad_enabled(False)
device = torch.device('cuda')
runner = FreeSplatterRunner(device)


_HEADER_ = '''
# FreeSplatter — Sparse-view Object Reconstruction (2DGS)
Feed **2–6 white-background views** of an object and get back a 3D Gaussian splat + mesh in seconds.
'''

_HELP_ = '''
💡 **Tips:**
- Input images should have a white background and a centred object. Enable *Remove background* if not.
- 2–6 views work best. More views → better coverage but diminishing returns.
- All views are assumed to share the same focal length.
'''


with gr.Blocks(analytics_enabled=False, title='FreeSplatter-O-2dgs', theme=gr.themes.Ocean()) as demo:
    gr.Markdown(_HEADER_)
    gr.Markdown(_HELP_)
    create_interface_views_to_3d(runner.run_views_to_3d)
    gr.Markdown('Model: [FreeSplatter-O-2dgs](https://huggingface.co/TencentARC/FreeSplatter)')


demo.queue().launch(
    share=False,
    server_name='0.0.0.0',
    server_port=41137,
    ssl_verify=False,
)
