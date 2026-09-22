# typed-decisions

```js
const Laya = require('typed-decisions')

const laya = new Laya()

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
