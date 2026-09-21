exports.makeBatch = function (items, padId, maxLength, padToMultiple) {
  const size = items.length
  const length = batchLength(items, maxLength, padToMultiple)
  const options = Math.max(2, ...items.map(item => item.markers.length))

  const inputIds = new Int32Array(size * length).fill(padId)
  const attention = []
  const markerPos = new Int32Array(size * options)
  const markerMask = []
  const qtype = new Int32Array(size)

  items.forEach((item, row) => {
    fillTokenRow(item, row, length, padId, inputIds, attention)
    fillMarkerRow(item, row, options, markerPos, markerMask)
    qtype[row] = item.qtype
  })

  return { size, length, options, inputIds, attention, markerPos, markerMask, qtype }
}

function batchLength (items, maxLength, padToMultiple) {
  const rawLength = Math.max(...items.map(item => item.ids.length), 1)
  if (!padToMultiple) return rawLength

  return Math.min(maxLength, Math.ceil(rawLength / padToMultiple) * padToMultiple)
}

function fillTokenRow (item, row, length, padId, inputIds, attention) {
  const rowAttention = new Array(length).fill(0)

  for (let i = 0; i < item.ids.length; i++) {
    inputIds[row * length + i] = item.ids[i]
    rowAttention[i] = 1
  }

  attention.push(rowAttention)
}

function fillMarkerRow (item, row, options, markerPos, markerMask) {
  const rowMask = new Array(options).fill(0)

  for (let i = 0; i < item.markers.length; i++) {
    markerPos[row * options + i] = item.markers[i]
    rowMask[i] = 1
  }

  markerMask.push(rowMask)
}
