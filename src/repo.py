import os

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Tree, Static, Pretty
from textual import on

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

class RepoInfo(Static):

    def __init__(self):
        super().__init__()
        # data_repo_path = os.path.join('..', 'data', 'repository')
        data_repo_path = '/home/julien/prog/apnea/test/data/repository'
        self.repository = Repository(data_repo_path)
        self.dataset_info = Pretty({})
        self.repo_tree = Tree("")
        self.repo_tree.root.expand_all()


    def _fill_tree(self):
        def add_node(name, node, sub_tree):
            if isinstance(sub_tree, dict):
                node.set_label(name)
                for key, value in sub_tree.items():
                    new_node = node.add("", data=key)
                    add_node(key, new_node, value)
            else:
                node.allow_expand = False
                node.set_label(name, data=name)

        datatsets = self.repository.metadata['datasets']
        list_node = []
        for ds_id in datatsets.keys():
            node = None
            if 'task_config' in datatsets[ds_id]:
                src_id = datatsets[ds_id]['task_config']['pipeline']['data']['dataset']['source']
                node = (ds_id, src_id)
            else:
                node = (ds_id, None)
            list_node.append(node)

        dict_tree = create_dict_tree(list_node, None)
        add_node("DataSets", self.repo_tree.root, dict_tree)


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
        self._fill_tree()

    def on_tree_node_highlighted(self, event:Tree.NodeSelected):
        n = event.node
        if n.data is not None:
            ds = self.repository.metadata['datasets'][n.data]
            self.dataset_info.update(ds)




class RepoApp(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.repo_info = RepoInfo()

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.repo_info
        yield Footer()


if __name__ == "__main__":
    app = RepoApp()
    app.run()