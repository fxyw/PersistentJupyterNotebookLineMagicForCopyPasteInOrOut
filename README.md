# PersistentJupyterNotebookLineMagicForCopyPasteInOrOut

make copy/past cell In[i] or Out[i] line magic function cc(i) or co(i) persistent

first run this in a jupyter notebook cell

```
%%file cccomagic.py
from IPython.display import Javascript,display

def cc(i):
  js_code = """
  // copy In [i] to the selected cell
  var selected = Jupyter.notebook.get_selected_index()-1; // shift-enter move cursor to cell below
  var cell = IPython.notebook.get_cell(selected);
  var cell_count = Jupyter.notebook.ncells();
  var i = Math.max(0, cell_count - 1);
  var ipn = Number(%s);
  var cells = Jupyter.notebook.get_cells();
  if(ipn>0){
    while (i>0 && cells[i].input_prompt_number != ipn){ i -= 1;}
  }else{
    i = selected + ipn;
  }
  //cell.code_mirror.doc.replaceSelection(cells[i].get_text());
  cell.set_text(cells[i].get_text());
  """ % (i)
  display(Javascript(js_code))

def co(i):
  js_code = """
  // copy output Out[i] from In [i] to the selected cell
  var selected = Jupyter.notebook.get_selected_index()-1; // shift-enter move cursor to cell below
  var cell = IPython.notebook.get_cell(selected);
  var cell_count = Jupyter.notebook.ncells();
  var i = Math.max(0, cell_count - 1);
  var ipn = Number(%s);
  var cells = Jupyter.notebook.get_cells();
  if(ipn>0){
    while (i>0 && cells[i].input_prompt_number != ipn){ i -= 1;}
  }else{
    i = selected + ipn;
  }
  //cell.code_mirror.doc.replaceSelection(cells[i].output_area.element[0].innerText); //will append not overwrite
  cell.set_text(cells[i].output_area.element[0].innerText);
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

find dir of the profile:
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

Restart jupyter notebook to make it work, using 

`%config InteractiveShellApp.extensions =['cccomagic']` 

to modify InteractiveShellApp.extensions have no effect.

The shortcut in 
 - [Jupyter notebook copy paste last In cell](https://web.archive.org/web/20200822013547/https://www.linkedin.com/pulse/jupyter-notbook-copy-paste-last-cell-wang-frank)
 - [Jupyter notebook insert output from above and more](https://web.archive.org/web/20200904003015/https://www.linkedin.com/pulse/jupyter-notebook-insert-output-from-above-more-wang-frank) 

can also match the cc fuction:
- ctrl-l to `cc -1` 
- shfit-l to `cc -2`
- alt-l to `cc -3`

We used `Jupyter.notebook.get_selected_index()-2` for Insert Input from three cells Above, alt-l. We can use ctrl-a, ctrl-c, ctrl-v to achieve `cc i`, with more key strokes, or use In[i] shift-enter (the later one will remove indent and mess the display up). 

Also note that the
 
`var cell = Jupyter.notebook.get_cell(Jupyter.notebook.get_selected_index())` 

way of select current cell will point to the cell below after run by shift-enter, and not overwrite the real current cell.
We updated cc(i) co(i) to use negative i for the most recent cells, or cells without input prompt number.

