// this is for the floating action button to be initialized when the html is loaded in the document
document.addEventListener('DOMContentLoaded', () => {
  const elems = document.querySelectorAll('.fixed-action-btn');
  // const instances = 
  M.FloatingActionButton.init(elems, {
    direction: 'left'
  });
});
