// home.js
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('settingsForm');
  const demoBtn = document.getElementById('demoBtn');

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    saveSettingsAndGo();
  });

  demoBtn.addEventListener('click', () => {
    // demo defaults
    document.getElementById('username').value = 'Player';
    document.getElementById('chooseX').checked = true;
    document.getElementById('difficulty').value = 'easy';
    document.getElementById('evaluation').value = 'heuristic';
    saveSettingsAndGo();
  });

  function saveSettingsAndGo() {
    const username = document.getElementById('username').value.trim() || 'Player';
    const symbol = document.querySelector('input[name="symbol"]:checked').value;
    const difficulty = document.getElementById('difficulty').value;
    const evaluation = document.getElementById('evaluation').value;

    const settings = { username, symbol, difficulty, evaluation };
    // store in sessionStorage for passing to game page
    sessionStorage.setItem('ttt_settings', JSON.stringify(settings));
    // go to game page
    window.location.href = 'game.html';
  }
});
