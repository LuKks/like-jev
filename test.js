const test = require('brittle')
const Laya = require('./index.js')

test('uses numeric question type for fallback temperature', async function (t) {
  t.plan(1)

  const laya = new Laya({
    engine: {
      name: 'laya',
      padToMultiple: null,
      tokenizer: {
        encode: function () {
          return { ids: [] }
        }
      },
      ids: { cls: 1, sep: 2, mask: 3, pad: 0, maskToken: '[MASK]' },
      config: {
        head_max_len: 32,
        max_len: 64,
        temperature: [1, 2, 3],
        temperature_by_options: {}
      },
      async forward () {
        return { logits: [[0, 1]], actProbabilities: [[1]] }
      },
      async close () {}
    }
  })

  const result = await laya.ask({}, {
    severity: {
      type: 'score',
      instructions: 'How severe is this?',
      criteria: ['low', 'high']
    }
  })

  t.ok(Math.abs(result.answers.severity.probabilities['1'] - 0.6224593312018546) < 1e-12)
})

test('English Core ML model answers typed questions', { timeout: 60000 * 10, skip: process.platform !== 'darwin' }, async function (t) {
  await realModelTest(t, { backend: 'coreml', model: 'english' }, 'laya-coreml-english')
})

test('multilingual Core ML model answers typed questions', { timeout: 60000 * 10, skip: process.platform !== 'darwin' }, async function (t) {
  await realModelTest(t, { backend: 'coreml', model: 'multilingual' }, 'laya-coreml-multilingual')
})

test('English ONNX model answers typed questions', { timeout: 60000 * 10 }, async function (t) {
  await realModelTest(t, { backend: 'onnx', model: 'english' }, 'laya-onnx-english')
})

test.solo('debug', { timeout: 60000 * 5 }, async function (t) {
  const laya = new Laya({ backend: 'coreml', model: 'multilingual' })
  await laya.ready()

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
        instructions: 'Does this require urgent attention?',
        criteria: {
          true: 'The issue needs immediate attention.',
          false: 'The issue can be handled through the normal queue.'
        }
      }
    }
  )

  console.log(JSON.stringify(result, null, 2))

  await laya.close()
})

async function realModelTest (t, opts, model) {
  const laya = new Laya(opts)

  await laya.ready()

  try {
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

    t.is(result.model, model)
    t.alike(Object.keys(result.answers), ['team', 'urgency', 'urgent'])
    t.is(result.answers.team.type, 'choice')
    t.ok(['billing', 'support'].includes(result.answers.team.choice))
    t.is(result.answers.urgency.type, 'score')
    t.ok(result.answers.urgency.score >= 0 && result.answers.urgency.score <= 2)
    t.is(result.answers.urgent.type, 'noul')
    t.ok(result.answers.urgent.noul >= 0 && result.answers.urgent.noul <= 1)
    t.ok(result.usage.input_tokens > 0)
    t.is(result.usage.output_tokens, 0)
  } finally {
    await laya.close()
  }
}
