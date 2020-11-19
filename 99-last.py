#%automagic 1
get_ipython().run_line_magic('automagic', '1')
from IPython.display import Javascript,display
from IPython.core.magic import register_line_magic

@register_line_magic
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

@register_line_magic
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
  
@register_line_magic
def ec(i):
    """execute cell In[i]"""
    cc(i)
    display(Javascript('IPython.notebook.execute_cell_range(IPython.notebook.get_selected_index()-1, IPython.notebook.get_selected_index())'))  
 
s = """
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
"""
get_ipython().run_cell_magic('js', '', s)