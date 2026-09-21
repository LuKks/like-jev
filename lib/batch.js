'use strict'

exports.makeBatch = function (items, padId, maxLength, padToMultiple) {
  const size = items.length
  const rawLength = Math.max(...items.map(item => item.ids.length), 1)
  const length = padToMultiple
    ? Math.min(maxLength, Math.ceil(rawLength / padToMultiple) * padToMultiple)
    : rawLength
  const options = Math.max(2, ...items.map(item => item.markers.length))
  const inputIds = new Int32Array(size * length).fill(padId)
  const attention = []
  const markerPos = new Int32Array(size * options)
  const markerMask = []
  const qtype = new Int32Array(size)

  items.forEach((item, row) => {
    const rowAttention = new Array(length).fill(0)
    for (let i = 0; i < item.ids.length; i++) {
      inputIds[row * length + i] = item.ids[i]
      rowAttention[i] = 1
    }
    attention.push(rowAttention)
    const rowMask = new Array(options).fill(0)
    for (let i = 0; i < item.markers.length; i++) {
      markerPos[row * options + i] = item.markers[i]
      rowMask[i] = 1
    }
    markerMask.push(rowMask)
    qtype[row] = item.qtype
  })

  return { size, length, options, inputIds, attention, markerPos, markerMask, qtype }
}
