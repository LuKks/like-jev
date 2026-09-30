const test = require('require-npm-global')('brittle')
const Jev = require('./index.js')

test('uses numeric question type for fallback temperature', async function (t) {
  t.plan(1)

  const jev = new Jev({
    engine: {
      backend: 'test',
      model: 'mock',
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

  const result = await jev.ask({}, {
    severity: {
      type: 'score',
      instructions: 'How severe is this?',
      criteria: ['low', 'high']
    }
  })

  t.ok(Math.abs(result.answers.severity.probabilities['1'] - 0.6224593312018546) < 1e-12)
})

test('throws coded errors when inputs exceed token limits', async function (t) {
  t.plan(5)

  const jev = new Jev({
    engine: {
      backend: 'test',
      model: 'mock',
      padToMultiple: null,
      tokenizer: {
        encode: function (text) {
          return { ids: Array.from({ length: text.length }, () => 1) }
        }
      },
      ids: { cls: 1, sep: 2, mask: 3, pad: 0, maskToken: '[MASK]' },
      config: {
        head_max_len: 64,
        max_len: 96,
        temperature: [1, 1, 1],
        temperature_by_options: {}
      },
      async forward () {
        return { logits: [[0, 1]], actProbabilities: [[1]] }
      },
      async close () {}
    }
  })

  let error

  try {
    await jev.ask('s', { q: { type: 'noul', instructions: 'urgent?', criteria: { true: 'x'.repeat(60), false: 'no' } } })
  } catch (err) {
    error = err
  }

  t.is(error && error.code, 'OPTION_TOO_LONG')
  t.ok(error && error.message.includes('question "q"'))

  try {
    await jev.ask('s', {
      q: {
        type: 'choice',
        instructions: 'pick',
        criteria: { a: 'a'.repeat(20), b: 'b'.repeat(20), c: 'c'.repeat(20) }
      }
    })
  } catch (err) {
    error = err
  }

  t.is(error && error.code, 'HEAD_TOO_LONG')

  try {
    await jev.ask('s', { q: { type: 'noul', instructions: 'i'.repeat(30), criteria: { true: 'yes', false: 'no' } } })
  } catch (err) {
    error = err
  }

  t.is(error && error.code, 'INSTRUCTIONS_TOO_LONG')

  try {
    await jev.ask('s'.repeat(60), { q: { type: 'noul', instructions: 'urgent?', criteria: { true: 'yes', false: 'no' } } })
  } catch (err) {
    error = err
  }

  t.is(error && error.code, 'STATE_TOO_LONG')
})

test.skip('real models answer typed questions', { timeout: 60000 * 30 }, async function (t) {
  const skip = process.platform !== 'darwin'

  const models = [
    { opts: { backend: 'onnx', model: 'english', precision: 'fp16' }, name: 'jev-onnx-english' },
    { opts: { backend: 'onnx', model: 'english', precision: 'fp32' }, name: 'jev-onnx-english' },
    { opts: { backend: 'onnx', model: 'multilingual', precision: 'fp16' }, name: 'jev-onnx-multilingual' },
    { opts: { backend: 'onnx', model: 'multilingual', precision: 'fp32' }, name: 'jev-onnx-multilingual' },
    { opts: { backend: 'coreml', model: 'english', precision: 'fp16' }, name: 'jev-coreml-english', skip },
    { opts: { backend: 'coreml', model: 'multilingual', precision: 'e8' }, name: 'jev-coreml-multilingual', skip },
    { opts: { backend: 'coreml', model: 'multilingual', precision: 'fp16' }, name: 'jev-coreml-multilingual', skip },

    { opts: { backend: 'onnx', model: 'multilingual', precision: 'fp16', device: 'cpu' }, name: 'jev-onnx-english' },
    { opts: { backend: 'onnx', model: 'multilingual', precision: 'fp16', device: 'webgpu' }, name: 'jev-onnx-english' },
    { opts: { backend: 'onnx', model: 'multilingual', precision: 'fp16', device: 'coreml' }, name: 'jev-onnx-english' },
  ]

  const active = models.filter(model => !model.skip)

  t.plan(active.length * 10)

  for (const model of active) {
    t.comment('Creating instance', model.opts.backend + '/' + model.opts.model + '/' + model.opts.precision + '/' + (model.opts.device || 'default'))

    const time = Date.now()

    const jev = new Jev(model.opts)

    await jev.ready()

    t.comment('jev.ready() time:', Date.now() - time, 'with', model.opts.backend + '/' + model.opts.model + '/' + model.opts.precision + '/' + (model.opts.device || 'default'))

    let result = null

    for (let i = 0; i < 10; i++) {
      const time = Date.now()

      const once = !result

      result = await jev.ask(
        {
          subject: 'Refund not received (#' + i + ')',
          body: 'The customer cancelled two weeks ago and still has no refund. (#' + i + ')'
        },
        {
          team: {
            type: 'choice',
            instructions: 'Which team should handle this? (#' + i + ')',
            criteria: {
              billing: 'payments and refunds',
              support: 'product help and bugs'
            }
          },
          urgency: {
            type: 'score',
            instructions: 'How urgent is this? (#' + i + ')',
            criteria: ['low', 'medium', 'high']
          },
          urgent: {
            type: 'noul',
            instructions: 'Does this require urgent attention? (#' + i + ')',
            criteria: {
              true: 'The issue needs immediate attention.',
              false: 'The issue can be handled through the normal queue.'
            }
          }
        }
      )

      if (once) {
        t.comment(JSON.stringify(result, null, 2))
      }

      t.comment('jev.ask(...) time:', Date.now() - time, 'ms with', model.opts.backend + '/' + model.opts.model + '/' + model.opts.precision + '/' + (model.opts.device || 'default'))
    }

    t.is(result.model, model.name)
    t.alike(Object.keys(result.answers), ['team', 'urgency', 'urgent'])
    t.is(result.answers.team.type, 'choice')
    t.ok(['billing', 'support'].includes(result.answers.team.choice))
    t.is(result.answers.urgency.type, 'score')
    t.ok(result.answers.urgency.score >= 0 && result.answers.urgency.score <= 2)
    t.is(result.answers.urgent.type, 'noul')
    t.ok(result.answers.urgent.noul >= 0 && result.answers.urgent.noul <= 1)
    t.ok(result.usage.input_tokens > 0)
    t.is(result.usage.output_tokens, 0)

    await jev.close()
  }
})

test.skip('debug', { timeout: 60000 * 5 }, async function (t) {
  const jev = new Jev({ backend: 'coreml', model: 'multilingual' })
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

  console.log(JSON.stringify(result, null, 2))

  await jev.close()
})
