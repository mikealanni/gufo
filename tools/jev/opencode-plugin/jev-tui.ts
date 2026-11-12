// TUI-side /jev: runs the control logic locally and shows the result in a dialog, so no model turn is started.
import { homedir } from "node:os";
import { join } from "node:path";
import { Plugin } from "@opencode/plugin/tui";
import { control } from "./ctl.ts";

const LOG_DIR = process.env.JEV_LOG_DIR ?? join(homedir(), ".gufo", "jev-log");

export default Plugin.define({
  id: "jev.tui",
  async setup(context: any) {
    // The server process loads every file in plugins/ too; only the TUI context has a keymap.
    if (!context?.keymap?.layer) return;
    context.keymap.layer(() => ({
      mode: "global",
      priority: 10,
      commands: [{
        id: "jev.control",
        title: "jev control",
        group: "jev",
        palette: true,
        slash: { name: "jev", arguments: true },
        enabled: () => true,
        run: async (input?: string) => {
          let text = String(input ?? "").trim() || "status";
          // Per-session pause needs a session id the TUI plugin does not have: act on all sessions.
          if (text === "pause" || text === "resume") text += " all";
          let out: string;
          try { out = await control(LOG_DIR, "tui", text); } catch (err) { out = `jev error: ${String(err)}`; }
          try { await context.ui.dialog.alert({ title: "jev", message: out }); }
          catch { context.ui.toast.show({ title: "jev", message: out, variant: "info", duration: 15000 }); }
        },
      }],
      bindings: ["jev.control"],
    }));
  },
});
