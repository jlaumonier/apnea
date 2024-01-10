from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Tree


class StopwatchApp(App):
    """A Textual app to manage stopwatches."""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        tree = Tree('Repository')
        node1 = tree.root.add('001')
        node2 = tree.root.add('002')
        yield Header()
        yield Footer()
        yield tree

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark


if __name__ == "__main__":
    app = StopwatchApp()
    app.run()