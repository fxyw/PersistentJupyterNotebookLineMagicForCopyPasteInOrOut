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
- alt-l to `cc -2`
- ctrl-alt-l to `cc -3`

We used `Jupyter.notebook.get_selected_index()-2` for Insert Input from two cells Above, alt-l.
```
%%js
Jupyter.keyboard_manager.edit_shortcuts.add_shortcut('ctrl-l', {
    help : 'Insert Input from Above',
    help_index : 'zz',
    handler: function(env) {
        var cm=env.notebook.get_selected_cell().code_mirror;
        cm.doc.replaceSelection(Jupyter.notebook.get_cell(Jupyter.notebook.get_selected_index()-1).get_text());
        cm.execCommand('goLineEnd');
        return false;
    }}
);
Jupyter.keyboard_manager.edit_shortcuts.add_shortcut('alt-l', {
    help : 'Insert Input from two cells above',
    help_index : 'zz',
    handler: function(env) {
        var cm=env.notebook.get_selected_cell().code_mirror;
        cm.doc.replaceSelection(Jupyter.notebook.get_cell(Jupyter.notebook.get_selected_index()-2).get_text());
        cm.execCommand('goLineEnd');
        return false;
    }}
);
Jupyter.keyboard_manager.edit_shortcuts.add_shortcut('ctrl-shift-l', {
    help : 'Insert Output from Above',
    help_index : 'zz',
    handler: function(env) {
        var cm=env.notebook.get_selected_cell().code_mirror;
        cm.doc.replaceSelection(Jupyter.notebook.get_cell(Jupyter.notebook.get_selected_index()-1).output_area.element[0].innerText);
        cm.execCommand('goLineEnd');
        return false;
    }}
);
Jupyter.keyboard_manager.edit_shortcuts.add_shortcut('ctrl-alt-l', {
    help : 'Insert Input from three cells Above',
    help_index : 'zz',
    handler: function(env) {
        var cm=env.notebook.get_selected_cell().code_mirror;
        cm.doc.replaceSelection(Jupyter.notebook.get_cell(Jupyter.notebook.get_selected_index()-3).get_text());
        cm.execCommand('goLineEnd');
        return false;
    }}
);
```
We can use ctrl-a, ctrl-c, ctrl-v to achieve `cc i`, with more key strokes, or use In[i] shift-enter (the later one will remove indent and mess the display up). 


Also note that the
 
`var cell = Jupyter.notebook.get_cell(Jupyter.notebook.get_selected_index())` 

way of select current cell will point to the cell below after run by shift-enter, and not overwrite the real current cell.
We updated cc(i) co(i) to use negative i for the most recent cells, or cells without input prompt number.

IPython magic command %recall or %rep fullfill %cc function for nonnegative input sequence, not inplace, and without arg co -1 in case the last cell have output, in other case the nearest cell with output, still not inplace.

Jupyter environment is uncertain for different installation methods and versions, or different servers or runs: the customized extension in .ipython\profile_default\ipython_config.py may work and also may not work, similarly for the shortcut definition that is saved to .jupyter\nbconfig\notebook.json. For a foolproof approach we can put these definitions in the .ipython\profile_default\startup folder, and in case it is not called in jupyter notebook initialization, run directly at the notebook first line, for example:

```%run -i C:/Users/frank.wang/.ipython/profile_default/startup/99-last.py```

User defined magic like %cc do not work without the % even if %automagic is 1.  