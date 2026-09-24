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

  t.is(result.answers.severity.probabilities['1'], 0.6225)
})

test.solo('answers typed questions with the real model (onnx)', { timeout: 60000 * 5 }, async function (t) {
  await realModelTest(t, { backend: 'onnx' }, 'laya')
})

test.solo('answers typed questions with the real model (mlx)', { timeout: 60000 * 5, skip: process.platform !== 'darwin' }, async function (t) {
  await realModelTest(t, { backend: 'mlx' }, 'laya-mlx')
})

test.solo('answers typed questions with the real model (coreml)', { timeout: 60000 * 10, skip: process.platform !== 'darwin' }, async function (t) {
  await realModelTest(t, { backend: 'coreml' }, 'laya-coreml')
})

test('backends agree on extracting the charged amount', { timeout: 60000 * 10 }, async function (t) {
  t.plan(4)

  const answers = await askEveryBackend(
    'Invoice #4471: the customer was charged $89.90 on March 3rd and again $89.90 on March 17th for the same order. The card statement shows a total of $179.80 for this order.',
    {
      total: {
        type: 'choice',
        instructions: 'What is the total amount charged for order #4471?',
        criteria: {
          '179.80': 'one hundred seventy-nine dollars and eighty cents',
          '89.90': 'eighty-nine dollars and ninety cents',
          '359.60': 'three hundred fifty-nine dollars and sixty cents'
        }
      }
    }
  )

  const probabilities = []

  for (const backend of Object.keys(answers)) {
    t.is(answers[backend].total.choice, '179.80')
    probabilities.push(answers[backend].total.probabilities['179.80'])
  }

  t.ok(Math.max(...probabilities) - Math.min(...probabilities) <= 0.15, 'probability drift across backends')
})

test('backends agree on routing a damaged-item refund', { timeout: 60000 * 10 }, async function (t) {
  t.plan(COMPARISON_BACKENDS.length * 2)

  const answers = await askEveryBackend(
    'The customer wants a refund for order #1023. The order arrived damaged on March 2nd and the customer sent photos. Policy: damaged items are refunded by the billing team within 14 days of delivery.',
    {
      team: {
        type: 'choice',
        instructions: 'Which team should handle this request?',
        criteria: {
          billing: 'refunds and payment issues',
          support: 'product help and usage questions',
          legal: 'contracts and liability'
        }
      }
    }
  )

  for (const backend of Object.keys(answers)) {
    t.is(answers[backend].team.choice, 'billing')
    t.ok(answers[backend].team.probabilities.billing > 0.98)
  }
})

test('backends agree on comparing prices', { timeout: 60000 * 10 }, async function (t) {
  t.plan(COMPARISON_BACKENDS.length)

  const answers = await askEveryBackend(
    'Product A costs $4.50 per unit and product B costs $3.75 per unit. The customer needs 10 units of one product.',
    { cheaper: { type: 'noul', instructions: 'Is product B the cheaper option for 10 units?' } }
  )

  for (const backend of Object.keys(answers)) {
    t.ok(answers[backend].cheaper.noul > 0.5)
  }
})

test('backends agree on classification with eight options', { timeout: 60000 * 10 }, async function (t) {
  t.plan(COMPARISON_BACKENDS.length * 2)

  const answers = await askEveryBackend(
    'The rocket launched successfully overnight, carrying a new crew of four astronauts to the orbital station. The mission will run science experiments for six months.',
    {
      topic: {
        type: 'choice',
        instructions: 'What is this text about?',
        criteria: {
          space: 'rockets, astronauts and orbital missions',
          sports: 'games, teams and athletes',
          finance: 'markets, stocks and earnings',
          politics: 'elections, laws and government',
          food: 'cooking, recipes and restaurants',
          travel: 'flights, hotels and vacations',
          health: 'medicine, symptoms and fitness',
          education: 'schools, courses and learning'
        }
      }
    }
  )

  for (const backend of Object.keys(answers)) {
    t.is(answers[backend].topic.choice, 'space')
    t.ok(answers[backend].topic.probabilities.space > 0.99)
  }
})

test('backends agree on a long state with mixed question types', { timeout: 60000 * 10 }, async function (t) {
  t.plan(COMPARISON_BACKENDS.length * 3)

  const answers = await askEveryBackend(
    'Support ticket #88231. Opened 2026-03-04 by customer j.doe@example.com. Plan: Business Pro, seats: 12, contract renews 2026-09-01. History: customer reports the export feature fails with error 500 every morning around 9am, blocking their daily reporting to stakeholders. They cleared cache, re-logged, tried two browsers and a different network. The API status page shows no incidents. Their data volume is about 4 GB, exports worked fine until the 2.1.0 update three days ago. Customer asks for a fix today because the report goes to the board tonight. No billing changes were requested and no refund was mentioned.',
    {
      queue: {
        type: 'choice',
        instructions: 'Which queue should this ticket go to?',
        criteria: {
          technical: 'bugs, errors and feature failures',
          billing: 'invoices, charges and refunds',
          sales: 'plans, upgrades and contracts',
          onboarding: 'account setup and training'
        }
      },
      urgency: {
        type: 'score',
        instructions: 'How urgent is this issue?',
        criteria: ['low: can wait days', 'medium: should be fixed today', 'high: needs immediate attention']
      },
      critical: { type: 'noul', instructions: 'Is the customer blocked from delivering their board report?' }
    }
  )

  for (const backend of Object.keys(answers)) {
    t.is(answers[backend].queue.choice, 'technical')
    t.ok(answers[backend].urgency.score >= 1.2, 'urgency above the low/medium midpoint')
    t.ok(answers[backend].critical.noul >= 0 && answers[backend].critical.noul <= 1)
  }
})

const COMPARISON_BACKENDS = process.platform === 'darwin' ? ['coreml', 'mlx', 'onnx'] : ['onnx']

// one engine resident at a time: all three together do not fit comfortably
// in memory on small machines and the onnx session dies under the pressure
async function askEveryBackend (state, questions) {
  const answers = {}

  for (const backend of COMPARISON_BACKENDS) {
    const laya = new Laya({ backend })
    try {
      await laya.ready()
      const result = await laya.ask(state, questions)
      answers[backend] = result.answers
    } finally {
      await laya.close()
    }
  }

  return answers
}

test('backends agree on transaction references across networks', { timeout: 60000 * 10 }, async function (t) {
  t.plan(COMPARISON_BACKENDS.length * 3)

  // full 64-char hashes tokenize to hundreds of byte-fallback tokens and
  // overflow the sequence, so real reports reference them by short suffix
  const answers = await askEveryBackend(
    [
      'Payment reconciliation for 2026-09-23.',
      'One incoming transfer today: Ethereum tx with reference e1d0f9b, 1.25 ETH, block 21403882, confirmed.',
      'Separately, a Bitcoin txid ending 5fbf5d48a moved 0.042 BTC and a Solana signature ending 6LxPn moved 300 SOL, both from yesterday.'
    ].join(' '),
    {
      network: {
        type: 'choice',
        instructions: 'Which network did the only transfer that arrived today come through?',
        criteria: { bitcoin: 'the BTC chain', ethereum: 'the ETH chain', solana: 'the SOL chain' }
      },
      quoted: {
        type: 'noul',
        instructions: 'Does the report mention the reference e1d0f9b?'
      },
      altered: {
        type: 'noul',
        instructions: 'Does the report mention the reference e1d0f9c?'
      }
    }
  )

  for (const backend of Object.keys(answers)) {
    t.is(answers[backend].network.choice, 'ethereum')
    t.ok(answers[backend].quoted.noul > 0.5, 'the exact reference is recognized')
    t.ok(answers[backend].altered.noul < 0.5, 'a one-character-altered reference is rejected')
  }
})

async function realModelTest (t, opts, model) {
  const laya = new Laya(opts)

  for (let i = 0; i < 5; i++ ) {
    console.time('laya.ask/' + model)

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

    console.timeEnd('laya.ask/' + model)
  }

  await laya.close()
}
