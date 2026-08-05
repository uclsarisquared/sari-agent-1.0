import base64
import re
import ast

from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalGroup, VerticalScroll, HorizontalGroup
from textual.widgets import (Markdown, LoadingIndicator, TextArea, Header,
                              RichLog, Button, Collapsible, Label)

import env
from openrouter import OpenRouterAgent, OpenRouterCfg

MAX_TIMESTEPS = 30
EXTRACTABLE_JSON = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

ACTION_MAP = {
    "MOVE_FWD":     lambda t: env.move_forward(t),
    "MOVE_BACK":    lambda t: env.move_backward(t),
    "MOVE_LEFT":    lambda t: env.move_left(t),
    "MOVE_RIGHT":   lambda t: env.move_right(t),
    "PAN_LEFT":     lambda t: env.pan_left(t),
    "PAN_RIGHT":    lambda t: env.pan_right(t),
    "TILT_UP":      lambda t: env.pan_up(t),
    "TILT_DOWN":    lambda t: env.pan_down(t),
    "EXTEND_LEFT":  lambda t: env.extend_left_hand_forward(t),
    "PULL_LEFT":    lambda t: env.pull_left_hand_backward(t),
    "EXTEND_RIGHT": lambda t: env.extend_right_hand_forward(t),
    "PULL_RIGHT":   lambda t: env.pull_right_hand_backward(t),
    "GRIP_LEFT":    lambda t: env.ToggleLeftGrip(),
    "GRIP_RIGHT":   lambda t: env.ToggleRightGrip(),
}


def get_state():
    state = env.RequestJson()
    screenshot = env.RequestScreenshot()
    return state, base64.b64encode(screenshot['image']).decode('utf-8')


def dispatch(actions, times):
    results = []
    for act, t in zip(actions, times):
        if act == "STOP":
            return True, results
        fn = ACTION_MAP.get(act)
        if fn:
            results.append((act, t, fn(t)))
        else:
            results.append((act, t, {"error": f"Unknown action: {act}"}))
    return False, results


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class TimestepDisplay(VerticalGroup):

    def __init__(self, timestep: int) -> None:
        self._timestep = timestep
        super().__init__()

    def on_mount(self) -> None:
        self.border_title = f"Timestep {self._timestep}"

    def compose(self) -> ComposeResult:
        with Collapsible(title="🤖 Model Response", collapsed=False):
            yield Markdown(classes="response_md")
        with Collapsible(title="⚙️ Actions", collapsed=False):
            yield RichLog(highlight=True, classes="action_log")

    def set_response(self, text: str) -> None:
        self.query_one(".response_md", Markdown).update(text)

    def log_action(self, action: str, times: int, result) -> None:
        self.query_one(".action_log", RichLog).write(f"{action} × {times} → {result}")

    def mark_stopped(self) -> None:
        self.border_title = f"Timestep {self._timestep} — STOP"


class StatusBar(HorizontalGroup):

    def compose(self) -> ComposeResult:
        yield Label("Mode: —", id="mode_label")
        yield Label("  |  Tokens: 0", id="token_label")

    def update_mode(self, mode: str) -> None:
        self.query_one("#mode_label", Label).update(f"Mode: {mode}")

    def update_tokens(self, tokens: int) -> None:
        self.query_one("#token_label", Label).update(f"  |  Tokens used: {tokens}")


class AgentLoop(VerticalGroup):

    BORDER_TITLE = "PantryPal Agent"

    def __init__(self, task: str, mode: str) -> None:
        self._task = task
        self._mode = mode
        super().__init__()

    def compose(self) -> ComposeResult:
        yield LoadingIndicator()

    def on_mount(self) -> None:
        self._run()

    def _finish(self) -> None:
        self.query_one(LoadingIndicator).display = False
        self.app.query_one(TaskInput).set_running(False)

    @work(thread=True)
    def _run(self) -> None:
        cfg = OpenRouterCfg(mode=self._mode)
        agent = OpenRouterAgent(cfg)

        for timestep in range(1, MAX_TIMESTEPS + 1):
            ts = TimestepDisplay(timestep)
            self.app.call_from_thread(self.mount, ts)

            try:
                state, image_b64 = get_state()
            except Exception as e:
                self.app.call_from_thread(ts.log_action, "ERROR", 0, {"error": str(e)})
                break

            try:
                response = agent.generate(
                    {"task": self._task, "state": state, "image": image_b64, "actions": []},
                    timestep,
                )
            except Exception as e:
                self.app.call_from_thread(ts.log_action, "ERROR", 0, {"error": str(e)})
                break

            self.app.call_from_thread(
                self.app.query_one(StatusBar).update_tokens,
                agent.metrics['total_tokens_used'],
            )

            # STOP returns plain string
            if isinstance(response, str):
                self.app.call_from_thread(ts.set_response, response)
                self.app.call_from_thread(ts.mark_stopped)
                break

            text = response['text']
            self.app.call_from_thread(ts.set_response, text)

            match = re.search(EXTRACTABLE_JSON, text)
            if not match:
                self.app.call_from_thread(ts.log_action, "PARSE ERROR", 0, {"error": "No JSON block in response"})
                break

            action_json = ast.literal_eval(match.group(1))
            actions = action_json.get('actions', [])
            times   = action_json.get('times', [])

            try:
                stopped, results = dispatch(actions, times)
            except Exception as e:
                self.app.call_from_thread(ts.log_action, "DISPATCH ERROR", 0, {"error": str(e)})
                break

            for act, t, result in results:
                self.app.call_from_thread(ts.log_action, act, t, result)

            if stopped:
                self.app.call_from_thread(ts.mark_stopped)
                break

        self.app.call_from_thread(self._finish)


class TaskInput(HorizontalGroup):

    def compose(self) -> ComposeResult:
        yield TextArea(placeholder="Enter task for PantryPal agent...", id="task_input")
        yield Button("inf_base", id="btn_base", variant="default")
        yield Button("inf_super", id="btn_super", variant="success")

    def set_running(self, running: bool) -> None:
        self.query_one("#btn_base", Button).disabled = running
        self.query_one("#btn_super", Button).disabled = running

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id not in ("btn_base", "btn_super"):
            return
        task = self.query_one("#task_input", TextArea).text.strip()
        if not task:
            return
        mode = "inf_base" if event.button.id == "btn_base" else "inf_super"
        self.set_running(True)
        self.app.query_one(StatusBar).update_mode(mode)
        self.app.query_one("#main_scroll", VerticalScroll).mount(AgentLoop(task, mode))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class PantryApp(App):

    TITLE = "PantryPal TUI"

    CSS = """
    Screen { layout: vertical; }

    #main_scroll { height: 1fr; }

    TimestepDisplay {
        border: solid $primary;
        margin: 1 0;
        padding: 0 1;
    }

    AgentLoop {
        border: solid $accent;
        margin: 1 0;
        padding: 0 1;
        height: auto;
    }

    TaskInput {
        height: 6;
        dock: bottom;
        padding: 1;
    }

    #task_input { width: 1fr; }

    StatusBar {
        height: 1;
        dock: bottom;
        background: $panel;
        padding: 0 1;
    }

    .response_md { padding: 0 1; }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="main_scroll")
        yield StatusBar()
        yield TaskInput()

    def on_mount(self) -> None:
        self.theme = "gruvbox"


if __name__ == "__main__":
    PantryApp().run()
