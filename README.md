# PersistentJupyterNotebookLineMagicForCopyPasteInOrOut

make copy/past cell In[i] or Out[i] line magic function cc(i) or co(i) persistent

first run this in a jupyter notebook cell

```
%%file cccomagic.py
from IPython.display import Javascript,display
def cc(i):
  js_code = """
  // copy In [i] to the selected cell
  var cell = Jupyter.notebook.get_cell(Jupyter.notebook.get_selected_index());
  var cell_count = Jupyter.notebook.ncells();
  var i = Math.max(0, cell_count - 1);
  var ipn = Number(%s);
  var cells = Jupyter.notebook.get_cells();
  while (i>0 && cells[i].input_prompt_number != ipn){ i -= 1;}
  cell.code_mirror.doc.replaceSelection(cells[i].get_text());
  """ % (i)
  display(Javascript(js_code))
def co(i):
  js_code = """
  // copy output Out[i] from In [i] to the selected cell
  var cell = Jupyter.notebook.get_cell(Jupyter.notebook.get_selected_index());
  var cell_count = Jupyter.notebook.ncells();
  var i = Math.max(0, cell_count - 1);
  var ipn = Number(%s);
  var cells = Jupyter.notebook.get_cells();
  while (i>0 && cells[i].input_prompt_number != ipn){ i -= 1;}
  cell.code_mirror.doc.replaceSelection(cells[i].output_area.element[0].innerText);
  """ % (i)
  display(Javascript(js_code))
def load_ipython_extension(ipython):
    """This function is called when the extension is
    loaded. It accepts an IPython InteractiveShell
    instance. We can register the magic with the
    `register_magic_function` method of the shell
    instance."""
    ipython.register_magic_function(cc, 'line')
    ipython.register_magic_function(co, 'line')
```    

then run this in the next cell:
```
!ipython profile create
```

find dir of the profilel:
```
!ipython locate profile 
```

open the following file from the dir, for e.g.:
```
"C:\home\yourname\.ipython\profile_default\ipython_config.py"
```

add the 3 lines:
```
c = get_config()
c.InteractiveShellApp.extensions = ['cccomagic']
c.InteractiveShell.automagic = True
```

Restart jupyter notebook to make it work, using %config to modify InteractiveShellApp.extensions have no effect.
run cc co will append to the cell below
