# like-jev

Jev-compatible typed decisions (choice, score, noul) with local Core ML and ONNX models.

```sh
npm i like-jev
```

## Usage

Runs offline after a one-time download, with no API key, rate limits, or per-token costs.

```js
import Jev from 'like-jev'

const jev = new Jev({ model: 'jev-multilingual-base' })

await jev.ready()

const response = await jev.ask({
  subject: 'Refund not received',
  body: 'The customer cancelled two weeks ago and still has no refund.'
}, {
  team: {
    type: 'choice',
    instructions: 'Which team should handle this?',
    criteria: {
      billing: 'Payments and refunds',
      technical: 'Bugs and integrations',
      sales: 'Pricing and new accounts'
    }
  },
  urgency: {
    type: 'score',
    instructions: 'How urgent is this?',
    criteria: ['low', 'medium', 'high']
  },
  urgent: {
    type: 'noul',
    instructions: 'Does this require urgent attention?',
    criteria: {
      true: 'Explicitly time-sensitive',
      false: 'No urgency expressed'
    }
  }
})

console.log(response.answers.team.choice) // => 'billing'
console.log(response.answers.urgency.score) // => 1.40 (medium-high)
console.log(response.answers.urgent.noul) // => 0.98 (true)

await jev.close()
```

## API

#### `jev = new Jev([options])`

Creates a model instance and starts loading it.

Model details:

- `jev-multilingual-base` = mmBERT-base (322M)

The Core ML backend is blazing fast for Apple devices.

Options:

```js
{
  // Defaults to Core ML on macOS and ONNX elsewhere.
  backend: 'auto' | 'onnx' | 'coreml',
  model: 'jev-multilingual-base'
}
```

Extra options for the `coreml` backend:

```js
{
  lengths: [128, 512]
}
```

Extra options for the `onnx` backend:

```js
{
  device: 'webgpu' | 'cpu' | 'cuda'
}
```

#### `await jev.ready()`

Resolves when the model and tokenizer are ready for use.

#### `response = await jev.ask(state, questions)`

Runs typed questions against `state` and returns answers keyed by question id.

- `state`: a string, object, or array.
- `questions`: a map of question ids to `noul`, `choice`, or `score` questions.

```js
const response = await jev.ask('My card was charged twice. Please help ASAP.', {
  team: {
    type: 'choice',
    instructions: 'Which team should handle this?',
    criteria: { billing: 'payments and refunds', support: 'product help and bugs' }
  },
  urgency: {
    type: 'score',
    instructions: 'How urgent is this?',
    criteria: ['low', 'medium', 'high']
  },
  urgent: {
    type: 'noul',
    instructions: 'Is this urgent?',
    criteria: { true: 'time-sensitive', false: 'no urgency stated' }
  }
})

console.log(response.answers) /* => {
  team: {
    type: 'choice',
    choice: 'billing',
    probabilities: { billing: 1, support: 0 },
    confidence: 1
  },
  urgency: {
    type: 'score',
    score: 1.4,
    legend: { '0': 'low', '1': 'medium', '2': 'high' },
    probabilities: { '0': 0.1, '1': 0.4, '2': 0.5 },
    confidence: 0.14
  },
  urgent: {
    type: 'noul',
    noul: 0.98
  }
} */
```

#### `response = await jev.noul(state, instructions, criteria)`

Send one `noul` question. The returned answer has `type` and `noul` properties.

```js
const response = await jev.noul(
  'My card was charged twice. Please help ASAP.',
  'Does this message convey urgency?',
  {
    true: 'Explicitly time-sensitive',
    false: 'No urgency expressed'
  }
)

console.log(response) /* => {
  type: 'noul',
  noul: 0.98
} */
```

#### `response = await jev.choice(state, instructions, criteria)`

Send one `choice` question. The returned answer has `type`, `choice`, `probabilities`, and `confidence` properties.

```js
const response = await jev.choice(
  'My card was charged twice. Please help ASAP.',
  'Which team should handle this?',
  {
    billing: 'Payments and refunds',
    technical: 'Bugs and integrations',
    sales: 'Pricing and new accounts'
  }
)

console.log(response) /* => {
  type: 'choice',
  choice: 'billing',
  probabilities: { billing: 1, technical: 0, sales: 0 },
  confidence: 1
} */
```

#### `response = await jev.score(state, instructions, criteria)`

Send one `score` question. The returned answer has `type`, `score`, `legend`, `probabilities`, and `confidence` properties.

```js
const response = await jev.score(
  'My card was charged twice. Please help ASAP.',
  'How frustrated is this customer?',
  ['Calm', 'Frustrated', 'Very angry']
)

console.log(response) /* => {
  type: 'score',
  score: 1.02,
  legend: { '0': 'Calm', '1': 'Frustrated', '2': 'Very angry' },
  probabilities: { '0': 0, '1': 0.98, '2': 0.02 },
  confidence: 0.91
} */
```

#### `await jev.close()`

Closes the model.

## Errors

```js
try {
  await jev.ask('My card was charged twice.', {})
} catch (err) {
  console.error(err.name, err.code, err.message)
}
```

The `code` property is set for these errors:

- `INVALID_STATE`: state must be a string, object, or array.
- `QUESTIONS_REQUIRED`: `ask()` needs at least one question.
- `UNKNOWN_QUESTION_TYPE`: the question type is not `choice`, `score`, or `noul`.
- `INVALID_CHOICE_CRITERIA`: `choice` criteria must be a nonempty map.
- `INVALID_SCORE_CRITERIA`: `score` criteria must be a nonempty array.
- `OPTION_TOO_LONG`: an answer option exceeds the token limit.
- `HEAD_TOO_LONG`: answer options leave too little room for instructions.
- `INSTRUCTIONS_TOO_LONG`: instructions exceed the available token budget.
- `STATE_TOO_LONG`: state exceeds the available token budget.

## License

MIT
