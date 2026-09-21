Yes—but the `.onnx` file alone is not enough.

The model expects tokenized tensors:

- `input_ids`
- `attention_mask`
- `marker_pos`
- `marker_mask`
- `qtype`

You also need:

- `laya.onnx` + `laya.onnx.data`
- `laya_config.json`
- the tokenizer files

The minimal direct dependencies are:

```bash
npm install onnxruntime-node @huggingface/tokenizers
```

Then the core is:

```js
import * as ort from 'onnxruntime-node'
import { Tokenizer } from '@huggingface/tokenizers'

const tokenizer = await Tokenizer.fromFile('./model/tokenizer/tokenizer.json')
const session = await ort.InferenceSession.create('./model/laya.onnx')

const result = await session.run({
  input_ids: new ort.Tensor('int64', inputIds, [1, length]),
  attention_mask: new ort.Tensor('int64', attentionMask, [1, length]),
  marker_pos: new ort.Tensor('int64', markerPos, [1, options]),
  marker_mask: new ort.Tensor('bool', markerMask, [1, options]),
  qtype: new ort.Tensor('int64', qtype, [1])
})

console.log(result.logits.data)
```

The difficult part is constructing the exact sequence:

```text
[CLS] question [SEP] [MASK] option [MASK] option ... [SEP] state [SEP]
```

and then applying the model’s temperature calibration and softmax.

So `@receptron/laya` is actually fairly thin: only `onnxruntime-node` and `@huggingface/tokenizers`. The direct implementation would mainly mean copying its sequence-building and output-decoding logic.

My recommendation: keep the direct runner in a separate file such as `laya.js`, not in the pure `index.js`. The useful source to copy is [`src/sequence.ts`](https://github.com/receptron/laya/blob/main/src/sequence.ts); the actual ONNX call is in [`src/laya.ts`](https://github.com/receptron/laya/blob/main/src/laya.ts).