// Minimal built-in declarations for the Phase 0 plugin (strict TS, no deps).
// Keeps `tsc --strict` green without installing @types/node.
declare module "node:fs" {
  export function readFileSync(path: string, enc?: string): any;
  export function writeFileSync(path: string, data: string): void;
  export function appendFileSync(path: string, data: string): void;
  export function mkdirSync(path: string, opts?: { recursive?: boolean }): void;
}
declare module "node:path" {
  export function basename(p: string): string;
  export function isAbsolute(p: string): boolean;
  export function join(...parts: string[]): string;
}
declare module "node:os" {
  export function homedir(): string;
}
declare module "node:crypto" {
  export function createHash(alg: string): { update(d: string): any; digest(enc?: string): any };
}
declare module "node:child_process" {
  export function spawnSync(cmd: string, args: string[], opts?: any): any;
}
declare module "bun:sqlite" {
  export class Database {
    constructor(path: string);
    exec(sql: string): void;
    prepare(sql: string): any;
  }
}
declare const process: { env: Record<string, string | undefined>; versions: Record<string, string>; exit(code?: number): never };
declare const performance: { now(): number };
