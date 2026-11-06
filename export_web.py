"""Export existing frozen checkpoint as little-endian FP32, plus QA fixtures."""
import hashlib,json
from pathlib import Path
import torch
from toy_demo import load_base, Experiment, FACTS, TRAIN_TEMPLATES, examples
root=Path(__file__).parent
out=root/'web'
model,tok=load_base(torch.device('cpu'))
weights=bytearray(); tensors={}
for name,t in model.state_dict().items():
    if name.startswith('memory.'): continue
    tensors[name]={'offset':len(weights)//4,'shape':list(t.shape)}
    weights.extend(t.detach().numpy().astype('<f4').tobytes())
(out/'model.bin').write_bytes(weights)
meta={'id':hashlib.sha256(weights).hexdigest(),'tensors':tensors,'vocabulary':tok.vocabulary,'probes':[TRAIN_TEMPLATES[k][0].format(s=s) for k,s,a in FACTS]}
(out/'model.json').write_text(json.dumps(meta,separators=(',',':')))
e=Experiment(model,tok,torch.device('cpu'))
questions=[q for q,_,_ in examples()+examples(True)]
logits,states=e.states(questions)
(root/'web-fixtures.json').write_text(json.dumps({'questions':questions,'logits':logits.tolist(),'states':states.tolist()}))
print('Exported',len(weights),'weight bytes')
