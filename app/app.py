import copy
import gradio as gr
from gradio_molecule3d import Molecule3D
import Bio
import Bio.SeqUtils

from utils.util_functions import merge_ranges
from predict import model_predict
from constants import *


def update_reps_based_on_radio(*args):
    struct, text = args[0], args[1]
    background, model, active_sites = args[2:4], args[4], args[5:]

    predicted_sites, confs, sequence = model_predict(model, struct, text)
    merged_sites = merge_ranges(predicted_sites, max_value=len(sequence))

    confidence_details = []
    new_reps = []

    # 1. cal summary
    summary_text = []
    for k, v in predicted_sites.items():
        if len(v) > 0:
            summary_text.append(f"{len(v)} {no_cat_dict[k]} site(s)")
    if len(summary_text) == 0:
        summary_text = ["No active sites identified."]
    summary_text = '; '.join(summary_text)

    # 2. cal dataframe
    detail_predicted_sites = {'b':[], '0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
    ass = []
    for k, v in predicted_sites.items():
        for vv in v:
            detail_predicted_sites[k].append(
                {'residue_type': sequence[vv-1], 'number': vv, 'confidence': confs[vv-1]}
            )
            ass.append(vv)
    for i in range(len(sequence)):
        if i+1 not in ass:
            detail_predicted_sites['b'].append(
                {'residue_type': sequence[i], 'number': i+1, 'confidence': confs[i]}
            )
    # 2.1 处理背景
    backgrounds = detail_predicted_sites.get('b', [])
    for r in backgrounds:
        confidence_details.append([
            'Background',
            Bio.SeqUtils.seq3(r['residue_type']).upper(),
            r['number'],
            r.get('confidence', 'N/A')
        ])
    # 2.2 处理活性位点
    for i in range(0, len(active_sites), 2):
        x, y = active_sites[i], active_sites[i+1]
        site_key = str(i//2)
        sites = detail_predicted_sites.get(site_key, [])
        for s in sites:
            confidence_details.append([
                no_cat_dict[site_key],
                Bio.SeqUtils.seq3(s['residue_type']).upper(),
                s['number'],
                s.get('confidence', 'N/A')
            ])

    # 3. cal reps
    # 3.1 background
    ranges = merged_sites['b']
    for r in ranges:
        old_reps = copy.deepcopy(default_reps)[0]
        old_reps['style'] = background[0][0].lower() + background[0][1:]
        old_reps['color'] = background[1][0].lower() + background[1][1:] + "Carbon"
        old_reps['residue_range'] = r
        new_reps.append(old_reps)
    # 3.2 active sites
    for i in range(0, len(active_sites), 2):
        x, y = active_sites[i], active_sites[i+1]
        ranges = merged_sites[str(i//2)]
        for r in ranges:
            old_reps = copy.deepcopy(default_reps)[0]
            old_reps['style'] = x[0].lower() + x[1:]
            old_reps['color'] = y[0].lower() + y[1:] + "Carbon"
            old_reps['residue_range'] = r
            new_reps.append(old_reps)

    return summary_text, confidence_details, Molecule3D(label="Identified Functional Sites", reps=new_reps)

def disable_fn(*x):
    return [gr.update(interactive=False)] * len(x)

def able_tip():
    return gr.update(visible=True)

def check_input(input):
    if input is not None:
        return gr.update(interactive=True)
    return gr.update(interactive=False)


with gr.Blocks(title="M3Site-app", theme=gr.themes.Default()) as demo:
    gr.Markdown("# M<sup>3</sup>Site: Leveraging Multi-Class Multi-Modal Learning for Accurate Protein Active Site Identification and Classification")
    gr.Markdown("""
    ## Overview
    **M<sup>3</sup>Site** is an advanced tool designed to accurately identify and classify protein active sites using a multi-modal learning approach. By integrating protein sequences, structural data, and functional annotations, M<sup>3</sup>Site provides comprehensive insights into protein functionality, aiding in drug design, synthetic biology, and understanding protein mechanisms.
    """)
    gr.Markdown("""
    ## How to Use
    1. **Select the Model**: Choose the pre-trained model for site prediction from the dropdown list.
    2. **Adjust Visual Settings**: Customize the visual style and color for background and active sites.
    3. **Upload Protein Structure**: Provide the 3D structure of the protein. You can upload from local or download from PDB Assym. Unit, PDB BioAssembly, AlphaFold DB, or ESMFold DB.
    4. **Enter Function Prompt**: Optionally provide a text description of the protein's function. If unsure, leave it blank.
    5. **Click "Predict"**: Hit the 'Predict' button to initiate the prediction. The predicted active sites will be highlighted in the structure visualization.
    6. **View Results**: The detailed results will be displayed below, including the identified active sites, their types, and confidence scores.
    """)

    with gr.Accordion("General Settings (Set before prediction)"):
        with gr.Row():
            model_drop = gr.Dropdown(model_list, label="Model Selection", value=model_list[0])
            gr.Markdown("")
            gr.Markdown("")
        with gr.Row():
            with gr.Row():
                style_dropb = gr.Dropdown(style_list, label="Style (Background)", value=style_list[0], min_width=1)
                color_dropb = gr.Dropdown(color_list, label="Color (Background)", value=color_list[0], min_width=1)
            with gr.Row():
                style_drop1 = gr.Dropdown(style_list, label="Style (CRI)", value=style_list[1], min_width=1)
                color_drop1 = gr.Dropdown(color_list, label="Color (CRI)", value=color_list[1], min_width=1)
            with gr.Row():
                style_drop2 = gr.Dropdown(style_list, label="Style (SCI)", value=style_list[1], min_width=1)
                color_drop2 = gr.Dropdown(color_list, label="Color (SCI)", value=color_list[2], min_width=1)
            with gr.Row():
                style_drop3 = gr.Dropdown(style_list, label="Style (PI)", value=style_list[1], min_width=1)
                color_drop3 = gr.Dropdown(color_list, label="Color (PI)", value=color_list[3], min_width=1)
        with gr.Row():
            with gr.Row():
                style_drop4 = gr.Dropdown(style_list, label="Style (PTCR)", value=style_list[1], min_width=1)
                color_drop4 = gr.Dropdown(color_list, label="Color (PTCR)", value=color_list[4], min_width=1)
            with gr.Row():
                style_drop5 = gr.Dropdown(style_list, label="Style (IA)", value=style_list[1], min_width=1)
                color_drop5 = gr.Dropdown(color_list, label="Color (IA)", value=color_list[5], min_width=1)
            with gr.Row():
                style_drop6 = gr.Dropdown(style_list, label="Style (SSA)", value=style_list[1], min_width=1)
                color_drop6 = gr.Dropdown(color_list, label="Color (SSA)", value=color_list[6], min_width=1)
            with gr.Row():
                gr.Markdown("")

        gr.Markdown('''
            *NOTE:* CRI indicates Covalent Reaction Intermediates, SCI indicates Sulfur-containing Covalent Intermediates, PI indicates Phosphorylated Intermediates, 
            PTCR indicates Proton Transfer & Charge Relay Systems, IA indicates Isomerization Activity, SSA indicates Substrate-specific Activities.
            ''')

    with gr.Row():
        gr.Markdown("<center><font size=5><b>Input Structure</b></font></center>")
        gr.Markdown("<center><font size=5><b>Output Predictions</b></font></center>")

    with gr.Row(equal_height=True):
        input_struct = Molecule3D(label="Input Protein Structure (Default Style)", reps=reps1)
        output_struct = Molecule3D(label="Output Protein Structure", reps=[])

    with gr.Row(equal_height=True):
        input_text = gr.Textbox(lines=1, label="Function Prompt", scale=16, min_width=1, placeholder="I don't know the function of this protein.")
        btn = gr.Button("Predict", variant="primary", scale=1, min_width=1, interactive=False)
        summary_output = gr.Label(label="", scale=18, min_width=1, show_label=False, elem_classes="info")

    gr.Markdown("### Result Details")
    confidence_output = gr.DataFrame(headers=["Active Site Type", "Residue Type", "Residue Number", "Confidence"])

    option_list = [
        style_dropb, color_dropb, model_drop, 
        style_drop1, color_drop1, 
        style_drop2, color_drop2, 
        style_drop3, color_drop3, 
        style_drop4, color_drop4, 
        style_drop5, color_drop5, 
        style_drop6, color_drop6
    ]

    tips = gr.Markdown("### *Tips: Please refresh the page to make a new prediction.*", visible=False)
    # gr.Markdown("## Citation")
    # gr.Markdown("If you find this tool helpful, please consider citing the following papers:")
    # with gr.Accordion("Citations", open=False):
    #     gr.Markdown('''```
    #                 @inproceedings{ouyangmmsite,
    #                     title={MMSite: A Multi-modal Framework for the Identification of Active Sites in Proteins},
    #                     author={Ouyang, Song and Cai, Huiyu and Luo, Yong and Su, Kehua and Zhang, Lefei and Du, Bo},
    #                     booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
    #                 }
    #                 @article{ouyangm3site,
    #                     title={M3Site: Leveraging Multi-Class Multi-Modal Learning for Accurate Protein Active Site Iden-tification and Classification},
    #                     author={Ouyang, Song and Luo, Yong and Su, Kehua and Zhang, Lefei and Du, Bo},
    #                     journal={xxxx},
    #                     year={xxxx},
    #                 }
    #                 ```''')
    
    # 绑定事件
    input_struct.change(check_input, inputs=input_struct, outputs=btn)
    btn.click(
        fn=able_tip, 
        inputs=[], 
        outputs=tips
    ).then(
        fn=disable_fn, 
        inputs=option_list, 
        outputs=option_list
    ).then(
        fn=update_reps_based_on_radio, 
        inputs=[input_struct, input_text] + option_list, 
        outputs=[summary_output, confidence_output, output_struct]
    ).then(
        fn=lambda x: x, 
        inputs=[input_struct], 
        outputs=[output_struct]
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=True)