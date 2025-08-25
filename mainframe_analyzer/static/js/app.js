document.getElementById('analyzeForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const cobol = document.getElementById('cobol').value;
  const res = await fetch('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cobol })
  });
  const json = await res.json();
  document.getElementById('output').textContent = JSON.stringify(json, null, 2);
});
