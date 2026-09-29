# typed-decisions

Requires macOS 14 or newer for Core ML. Core ML uses the English ModernBERT-large checkpoint on macOS. ONNX uses the same English checkpoint on Linux, Windows, and other systems. Set `backend: 'onnx'` or `backend: 'coreml'` to select a backend.

```js
const Laya = require('typed-decisions')

const laya = new Laya({ backend: 'onnx' })
```

Model files download on first use. Core ML uses fixed 128/512-token buckets. ONNX uses the English model's 512-token context.

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
