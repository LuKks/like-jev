# like-jev

Jev-compatible typed decisions (choice, score, noul) with local Core ML and ONNX models.

```sh
npm i like-jev
```

## Usage

Runs offline after a one-time download, with no API key, rate limits, or per-token costs.

```js
import Jev from 'like-jev'

const jev = new Jev({ model: 'english' })

await jev.ready()

const result = await jev.ask(
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
      instructions: 'Does this require urgent attention?',
      criteria: {
        true: 'The issue needs immediate attention.',
        false: 'The issue can be handled through the normal queue.'
      }
    }
  }
)

console.log(result.answers.team.choice)

await jev.close()
```

## API

### `jev = new Jev([options])`

Creates a model instance and starts loading it.

Options:

```js
{
  // 'auto' selects Core ML on macOS, and ONNX on other platforms.
  backend: 'auto' | 'onnx' | 'coreml',
  // The model 'multilingual' requires Core ML.
  model: 'english' | 'multilingual'
}
```

Extra options for the `coreml` backend:

```js
{
  // ... model details params
  model: 'english' | 'multilingual'
  // The model 'multilingual' supports precision 'e8' also.
  precision: 'fp16',
  // The model 'multilingual' supports length up to 1024 also.
  lengths: [128, 512]
}
```

Extra options for the `onnx` backend:

```js
{
  model: 'english', // The model 'multilingual' is not available for ONNX.
  logLevel: 'error',
  executionProviders: ['webgpu', 'cpu'],
  sessionOptions: {}
}
```

The ONNX model has a 512-token context.

### `await jev.ready()`

Resolves when the model and tokenizer are ready for use.

### `result = await jev.ask(state, questions)`

Runs one or more typed questions against `state` and returns the model name, answers, and token usage. `state` can be a string or a value that can be serialized as JSON. `questions` is an object whose keys identify the questions.

Each question has an `instructions` value and a `type`. Instructions can be a string or a JSON-serializable value.

A `choice` question selects one criterion. Its `criteria` can be an array of labels or an object that maps labels to descriptions. A `score` question returns a score based on the ordered criteria array. A `noul` question returns a value from `0` to `1`, where `0` represents false and `1` represents true. Its optional `criteria` object can provide `false` and `true` descriptions:

```js
{
  type: 'noul',
  instructions: 'Does this require urgent attention?',
  criteria: {
    false: 'No urgent action is needed',
    true: 'Urgent action is needed'
  }
}
```

The result contains the selected `model`, an `answers` object keyed by question ID, and token `usage`. The `model` name follows `jev-[backend]-[model]`, for example `jev-coreml-multilingual`. Choice answers include `choice` and label-keyed `probabilities`. Score answers include `score`, `legend`, and index-keyed `probabilities`. Noul answers include `noul`. Choice and score answers also include `confidence`. `output_tokens` is always `0`.

Throws an error if `questions` is empty, a question type is unknown, or an input exceeds an internal token limit. Limit errors carry a `code`: `OPTION_TOO_LONG` when one option exceeds 48 tokens, `HEAD_TOO_LONG` when the options leave too little room for instructions, `INSTRUCTIONS_TOO_LONG` when the instructions exceed the remaining head budget, or `STATE_TOO_LONG` when the state exceeds the remaining context.

### `await jev.close()`

Closes the model.

## License

MIT
