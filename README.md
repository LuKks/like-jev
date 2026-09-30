# typed-decisions

Requires macOS 14 or newer for Core ML. Core ML runs the English ModernBERT-large checkpoint by default and the multilingual mmBERT-base checkpoint with `model: 'multilingual'`. ONNX runs the English checkpoint on Linux, Windows, and other systems; the multilingual model is Core ML only. Set `backend: 'onnx'` or `backend: 'coreml'` to select a backend.

```js
const Laya = require('typed-decisions')

const laya = new Laya({ model: 'multilingual' })
```

Model files download on first use. Core ML uses fixed buckets: English 128/512, multilingual 128/512 by default with 256/1024 and the smaller `e8` weights available through `lengths` and `precision`. ONNX uses the English model's 512-token context.

```js
const result = await laya.ask(
  {
    subject: 'Refund not received',
    body: 'The customer cancelled two weeks ago and still has no refund.'
  },
  {
    team: {
      type: 'choice',
      instructions: 'Which team should handle this?',
      criteria: {
        billing: 'payments and refunds',
        support: 'product help and bugs'
      }
    },
    urgency: {
      type: 'score',
      instructions: 'How urgent is this?',
      criteria: ['low', 'medium', 'high']
    },
    urgent: {
      type: 'noul',
      instructions: 'Does this require urgent attention?'
    }
  }
)
```
