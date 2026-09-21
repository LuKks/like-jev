'use strict'

const fs = require('fs')
const fsp = require('fs/promises')
const os = require('os')
const path = require('path')
const { Readable } = require('stream')
const { pipeline } = require('stream/promises')

exports.cacheDir = function (name) {
  if (process.platform === 'darwin') return path.join(os.homedir(), 'Library', 'Caches', name)
  if (process.platform === 'win32') {
    const base = process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local')
    return path.join(base, name, 'Cache')
  }
  return path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), name)
}

exports.ensureBundle = async function (opts) {
  if (bundleComplete(opts.files, opts.cacheDir)) return opts.cacheDir
  if (opts.fallbackDir && bundleComplete(opts.files, opts.fallbackDir)) return opts.fallbackDir

  await fsp.mkdir(opts.cacheDir, { recursive: true })
  for (const file of opts.files) {
    const target = path.join(opts.cacheDir, file)
    if (fs.existsSync(target)) continue
    await download(opts.base, file, target)
  }

  return opts.cacheDir
}

function bundleComplete (files, dir) {
  if (!dir) return false
  return files.every(file => fs.existsSync(path.join(dir, file)))
}

exports.bundleComplete = bundleComplete

async function download (base, file, target) {
  const url = `${base}/${file.split('/').map(encodeURIComponent).join('/')}`
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not download ${file}: ${response.status} ${response.statusText}`)

  fs.mkdirSync(path.dirname(target), { recursive: true })
  const temporary = target + '.part'
  await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(temporary))
  await fsp.rename(temporary, target)
}
