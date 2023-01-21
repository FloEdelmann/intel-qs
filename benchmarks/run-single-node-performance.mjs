#!/usr/bin/env node

import { createReadStream, createWriteStream } from 'node:fs';
import { readdir, mkdir, readFile, rename } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { parse as parseCsv, format as formatCsv } from 'fast-csv';

const workingDirectory = new URL('./', import.meta.url);

const currentDate = new Date().toISOString().replace(/\..*/, '').replace(/:/g, '-').replace('T', '_');
const outputDirectory = new URL(`./output/${currentDate}_single_node_performance/`, workingDirectory);
const csvFile = new URL('./single_node_performance.csv', outputDirectory);
const csvFileWithPapi = new URL('./single_node_performance_with_papi.csv', outputDirectory);

const papiEvents = {
  'PAPI_L1_TCM': 'Level 1 cache misses',
  'PAPI_L2_TCM': 'Level 2 cache misses',
  'PAPI_L3_TCM': 'Level 3 cache misses',
  'PAPI_FP_INS': 'Floating point instructions',
  'PAPI_FP_OPS': 'Floating point operations',
  'PAPI_SP_OPS': 'Single precision floating point operations',
  'PAPI_DP_OPS': 'Double precision floating point operations',
}

await mkdir(outputDirectory, { recursive: true });

const numQubits = 30;
const numRepetitions = 3;

const childProcess = spawn(`./bin/single_node_performance.exe ${numQubits} ${numRepetitions}`, {
  cwd: workingDirectory,
  env: {
    ...process.env,
    PAPI_EVENTS: Object.keys(papiEvents).join(','),
  },
  shell: true,
});

let writeStream = createWriteStream(csvFile);
childProcess.stdout.pipe(writeStream);
childProcess.stdout.pipe(process.stdout);

childProcess.on('close', async () => {
  writeStream.close();

  const papiFolder = new URL('./papi_hl_output/', import.meta.url);
  const papiFiles = await readdir(papiFolder);

  const papiJsonFile = new URL(papiFiles[0], papiFolder);
  const papiData = JSON.parse(await readFile(papiJsonFile));

  await rename(papiFolder, new URL('./papi_hl_output/', outputDirectory));

  writeStream = createWriteStream(csvFileWithPapi);
  let rowCount = 0;

  createReadStream(csvFile)
    .pipe(parseCsv({
      headers: true,
      trim: true,
      comment: '[', // ignore MPI output
    }))
    .pipe(formatCsv({
      headers: true,
      transform: row => ({
        ...row,
        ...getPapiRowData(papiData, rowCount++),
      }),
    }))
    .pipe(writeStream);
});

const getPapiRowData = (papiData, rowCount) => {
  const papiRowData = papiData.threads[0].regions[rowCount];
  // console.log({ papiRowData });

  if (!papiRowData) {
    return {};
  }

  let regionName;
  let entries;

  if ('name' in papiRowData) {
    // new format
    regionName = papiRowData.name;
    entries = Object.entries(papiRowData);

    // console.log('new format', { regionName, entries });
  }
  else {
    // old format
    regionName = Object.keys(papiRowData)[0];
    entries = Object.entries(papiRowData[regionName]);

    // console.log('old format', { regionName, entries });
  }

  const filteredEntries = entries.filter(([key]) => key !== 'name' && key !== 'parent_region_id');
  // console.log({ filteredEntries });

  const mappedEntries = filteredEntries.map(([key, value]) => [key.trim().replace(/[:-]/g, '_'), value]);
  // console.log({ mappedEntries });

  return Object.fromEntries(mappedEntries);
}
