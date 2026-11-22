// A tiny stack virtual machine: runs every drawn program of the tiny stack ISA exactly as the
// published reference semantics define it, and writes stdout.bin and exit.txt per instance.
// Trap exit codes: 2 underflow, 3 overflow, 4 division by zero, 5 bad opcode, 6 memory,
// 7 call depth, 8 return on empty call stack, 9 pc or immediate outside, 10 steps, 11 output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const STACK_CAP = 256;
const CALLS_CAP = 64;
const MEM_WORDS = 256;
const STEP_CAP = 1_000_000;
const OUT_CAP = 65_536;
const INT32_MIN = -2147483648;

// Named opcodes packed as byte*1000 + immediateBytes*100 + pops*10 + pushes.
const OP_PACKED = [
  0, 1101, 2401, 3010, 4012, 5022, 6023, 7033, 16021, 17021, 18021, 19021, 20021, 21021, 22021,
  23011, 24021, 25021, 26021, 27011, 28021, 29021, 30021, 32021, 33021, 34021, 48200, 49210,
  50210, 51200, 52000, 64011, 65020, 80010, 81010, 82001, 83010,
];

const SHAPE_BY_BYTE = new Array(256);
for (const packedRow of OP_PACKED) {
  SHAPE_BY_BYTE[Math.floor(packedRow / 1000)] = [Math.floor(packedRow / 100) % 10, Math.floor(packedRow / 10) % 10, packedRow % 10];
}

// The binary operators ( a b -- r ) by opcode byte; a zero divisor traps before these run.
const BINOP = {
  0x10: (a, b) => (a + b) | 0,
  0x11: (a, b) => (a - b) | 0,
  0x12: (a, b) => Math.imul(a, b),
  0x13: (a, b) => (a === INT32_MIN && b === -1 ? a : (a / b) | 0),
  0x14: (a, b) => (b === -1 ? 0 : (a % b) | 0),
  0x15: (a, b) => Math.floor((a >>> 0) / (b >>> 0)) | 0,
  0x16: (a, b) => ((a >>> 0) % (b >>> 0)) | 0,
  0x18: (a, b) => a & b,
  0x19: (a, b) => a | b,
  0x1a: (a, b) => a ^ b,
  0x1c: (a, b) => a << (b & 31),
  0x1d: (a, b) => a >> (b & 31),
  0x1e: (a, b) => (a >>> (b & 31)) | 0,
  0x20: (a, b) => (a === b ? 1 : 0),
  0x21: (a, b) => (a < b ? 1 : 0),
  0x22: (a, b) => ((a >>> 0) < (b >>> 0) ? 1 : 0),
};

function runProgram(prog: Uint8Array, inp: Uint8Array) {
  const progLen = prog.length;
  const inLen = inp.length;
  const stack = new Int32Array(STACK_CAP);
  const calls = new Int32Array(CALLS_CAP);
  const mem = new Int32Array(MEM_WORDS);
  const out = new Uint8Array(OUT_CAP);
  let sp = 0;
  let csp = 0;
  let outLen = 0;
  let inCursor = 0;
  let pc = 0;
  let steps = 0;
  const done = (exitCode: number) => ({ outBytes: out.slice(0, outLen), exitCode });

  for (;;) {
    if (steps === STEP_CAP) return done(10);
    if (pc < 0 || pc >= progLen) return done(9);
    const op = prog[pc];
    const shape = SHAPE_BY_BYTE[op];
    if (shape === undefined) return done(5);
    const immLen = shape[0];
    const popCount = shape[1];
    const pushCount = shape[2];
    if (pc + immLen >= progLen) return done(9);
    steps += 1;
    if (sp < popCount) return done(2);
    if (sp - popCount + pushCount > STACK_CAP) return done(3);
    const nextPc = pc + 1 + immLen;
    pc = nextPc;

    switch (op) {
      case 0x00:
        return done(0);
      case 0x01:
        stack[sp] = (prog[nextPc - 1] << 24) >> 24;
        sp += 1;
        break;
      case 0x02:
        stack[sp] = prog[nextPc - 4] | (prog[nextPc - 3] << 8) | (prog[nextPc - 2] << 16) | (prog[nextPc - 1] << 24);
        sp += 1;
        break;
      case 0x03:
        sp -= 1;
        break;
      case 0x04:
        stack[sp] = stack[sp - 1];
        sp += 1;
        break;
      case 0x05: {
        const heldTop = stack[sp - 1];
        stack[sp - 1] = stack[sp - 2];
        stack[sp - 2] = heldTop;
        break;
      }
      case 0x06:
        stack[sp] = stack[sp - 2];
        sp += 1;
        break;
      case 0x07: { // ROT: ( a b c -- b c a )
        const heldDeep = stack[sp - 3];
        stack[sp - 3] = stack[sp - 2];
        stack[sp - 2] = stack[sp - 1];
        stack[sp - 1] = heldDeep;
        break;
      }
      case 0x17:
        stack[sp - 1] = -stack[sp - 1] | 0;
        break;
      case 0x1b:
        stack[sp - 1] = ~stack[sp - 1];
        break;
      case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16:
      case 0x18: case 0x19: case 0x1a: case 0x1c: case 0x1d: case 0x1e:
      case 0x20: case 0x21: case 0x22: {
        const bVal = stack[sp - 1];
        const aVal = stack[sp - 2];
        if (op === 0x13 || op === 0x14 || op === 0x15 || op === 0x16) {
          if (bVal === 0) return done(4);
        }
        sp -= 1;
        stack[sp - 1] = BINOP[op](aVal, bVal);
        break;
      }
      case 0x30: case 0x31: case 0x32: { // JMP / JZ / JNZ, signed relative offset
        const relOff = ((prog[nextPc - 2] | (prog[nextPc - 1] << 8)) << 16) >> 16;
        let jumpTaken = op === 0x30;
        if (op !== 0x30) {
          sp -= 1;
          const flagVal = stack[sp];
          jumpTaken = op === 0x31 ? flagVal === 0 : flagVal !== 0;
        }
        if (jumpTaken) pc = nextPc + relOff;
        break;
      }
      case 0x33:
        if (csp === CALLS_CAP) return done(7);
        calls[csp] = nextPc;
        csp += 1;
        pc = prog[nextPc - 2] | (prog[nextPc - 1] << 8);
        break;
      case 0x34:
        if (csp === 0) return done(8);
        csp -= 1;
        pc = calls[csp];
        break;
      case 0x40: {
        const addr = stack[sp - 1];
        if (addr < 0 || addr >= MEM_WORDS) return done(6);
        stack[sp - 1] = mem[addr];
        break;
      }
      case 0x41: { // STORE: ( v addr -- )
        const addr = stack[sp - 1];
        const heldVal = stack[sp - 2];
        if (addr < 0 || addr >= MEM_WORDS) return done(6);
        sp -= 2;
        mem[addr] = heldVal;
        break;
      }
      case 0x50: { // PRINT, decimal; a print that would cross the output limit writes nothing
        const printVal = stack[sp - 1];
        const magNow = printVal < 0 ? 0 - printVal : printVal;
        let magCount = magNow;
        let digitCount = 1;
        while (magCount >= 10) {
          magCount = Math.floor(magCount / 10);
          digitCount += 1;
        }
        const printLen = digitCount + (printVal < 0 ? 1 : 0);
        if (outLen + printLen > OUT_CAP) return done(11);
        sp -= 1;
        let writePos = outLen + printLen - 1;
        let magWrite = magNow;
        if (magWrite === 0) {
          out[writePos] = 48;
          writePos -= 1;
        }
        while (magWrite > 0) {
          out[writePos] = 48 + (magWrite % 10);
          writePos -= 1;
          magWrite = Math.floor(magWrite / 10);
        }
        if (printVal < 0) out[outLen] = 45;
        outLen += printLen;
        break;
      }
      case 0x51:
        if (outLen + 1 > OUT_CAP) return done(11);
        sp -= 1;
        out[outLen] = stack[sp] & 255;
        outLen += 1;
        break;
      case 0x52: { // IN, -1 once the input is used up, and every time after
        let fedByte = -1;
        if (inCursor < inLen) {
          fedByte = inp[inCursor];
          inCursor += 1;
        }
        stack[sp] = fedByte;
        sp += 1;
        break;
      }
      case 0x53:
        sp -= 1;
        return done(stack[sp] & 255);
    }
  }
}

const instanceName = /^i[0-9]{3}$/;

for (const entryName of readdirSync('/input').sort()) {
  if (instanceName.test(entryName)) {
    const outcome = runProgram(
      new Uint8Array(readFileSync(join('/input', entryName, 'program.bin'))),
      new Uint8Array(readFileSync(join('/input', entryName, 'input.bin'))),
    );
    mkdirSync(join('/output', entryName), { recursive: true });
    writeFileSync(join('/output', entryName, 'stdout.bin'), outcome.outBytes);
    writeFileSync(join('/output', entryName, 'exit.txt'), `${outcome.exitCode}\n`);
  }
}
