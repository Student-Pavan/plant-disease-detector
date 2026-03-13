/* Image preview + drag-and-drop for the upload form */
(function () {
  const dropZone    = document.getElementById('drop-zone');
  const fileInput   = document.getElementById('file-input');
  const preview     = document.getElementById('preview');
  const content     = document.getElementById('drop-zone-content');
  const submitBtn   = document.getElementById('submit-btn');
  const browseLink  = document.querySelector('.browse-link');

  if (!dropZone) return; // not on the upload page

  /* Open file dialog when clicking the drop zone or browse link */
  dropZone.addEventListener('click', () => fileInput.click());
  browseLink.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      showPreview(fileInput.files[0]);
    }
  });

  /* Drag-and-drop */
  ['dragenter', 'dragover'].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });
  dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      showPreview(file);
    }
  });

  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.classList.remove('hidden');
      content.classList.add('hidden');
      submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }
})();
