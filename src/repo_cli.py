import os

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Tree, Static, Pretty, Label, Button
from textual.screen import Screen
from textual import on, work

from pipeline.repository import Repository


# https://stackoverflow.com/questions/31643568/python-create-a-nested-dictionary-from-a-list-of-parent-child-values
# Thanks Thell
def create_dict_tree(node_map, root=None):
    """ Given a list of tuples (child, parent) return the nested dictionary representation. """
    def traverse(parent, node_map, seen):
        children = {}
        for edge in node_map:
            if edge[1] == parent and edge[0] not in seen:
                seen.add(edge[0])
                children[edge[0]] = traverse(edge[0], node_map, seen)
        return children

    return traverse(root, node_map, {root})


class QuestionScreen(Screen[bool]):
    """Screen with a parameter."""

    def __init__(self, question: str) -> None:
        self.question = question
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Label(self.question)
        yield Button("Yes", id="yes", variant="success")
        yield Button("No", id="no")

    @on(Button.Pressed, "#yes")
    def handle_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def handle_no(self) -> None:
        self.dismiss(False)


class RepoInfo(Static):

    def __init__(self):
        super().__init__()
        data_repo_path = os.path.join('test', 'data', 'repository')
        #data_repo_path = os.path.join('data', 'repository')
        self.repository = Repository(data_repo_path)
        self.dataset_info = Pretty({})
        self.repo_tree = Tree("")
        self.highlighted_node = None

    def fill_tree(self):
        def add_node(name, node, sub_tree):
            if isinstance(sub_tree, dict):
                node.set_label(name)
                for key, value in sub_tree.items():
                    new_node = node.add("", data=key)
                    add_node(key, new_node, value)
            else:
                node.allow_expand = False
                node.set_label(name, data=name)

        self.repo_tree.clear()
        list_nodes = self.repository.get_list_tree_chain()
        dict_tree = create_dict_tree(list_nodes, None)
        add_node("DataSets", self.repo_tree.root, dict_tree)
        self.repo_tree.root.expand_all()

    def get_task(self, id):
        ds = self.repository.metadata['datasets'][id]
        task_name = ''
        if 'task_config' in ds:
            task_name = str(ds['task_config']['pipeline']['data']['tasks']['task_func']['_target_'])
            task_name = task_name.split('.')[-1]
        return task_name

    def compose(self) -> ComposeResult:
        yield self.repo_tree
        yield self.dataset_info

    def _on_mount(self, event: events.Mount) -> None:
        self.fill_tree()

    def on_tree_node_highlighted(self, event:Tree.NodeSelected):
        n = event.node
        if n.data is not None:
            ds = self.repository.metadata['datasets'][n.data]
            self.dataset_info.update(ds)
            self.highlighted_node = n.data


class RepoApp(App):
    BINDINGS = [('d', 'delete_node', 'Delete highligted node')]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.repo_info = RepoInfo()

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.repo_info
        yield Footer()

    @work
    async def action_delete_node(self):
        if await self.push_screen_wait(QuestionScreen("Delete highlighted datasets ?"),):
            self.repo_info.repository.remove_dataset(self.repo_info.highlighted_node)
            self.repo_info.fill_tree()


if __name__ == "__main__":
    app = RepoApp()
    app.run()