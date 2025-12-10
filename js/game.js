// Tic-Tac-Toe AI 
// Pure Alpha-Beta, No Randomness, Difficulty Depends Only on Depth
// Supports: classical heuristic + ML evaluation (plug-in)
// GUI controlled by game.html

document.addEventListener('DOMContentLoaded', () => {
    /* ===================== State ===================== */
    let board = Array(9).fill(null);
    let playerSymbol = null;
    let aiSymbol = null;
    let gameOver = false;

    /* ===================== DOM ===================== */
    const boardEl = document.getElementById('board');
    const gameMessage = document.getElementById('gameMessage');
    const aiDecision = document.getElementById('aiDecision');
    const evalList = document.getElementById('evalList');
    const backBtn = document.getElementById('backBtn');
    const resetBtn = document.getElementById('resetBtn');
    const topStatus = document.getElementById('topStatus');

    /* ===================== Load Settings ===================== */
    const settings = JSON.parse(sessionStorage.getItem('ttt_settings') || '{}');
    playerSymbol = settings.symbol || 'X';
    aiSymbol = (playerSymbol === 'X') ? 'O' : 'X';
    topStatus.textContent = `${settings.username || 'Player'} — ${settings.difficulty || 'normal'} — ${settings.evaluation || 'heuristic'}`;

    /* ===================== Environment ===================== */
    function generateLegalMoves(b) {
        const moves = [];
        for (let i = 0; i < 9; i++) if (!b[i]) moves.push(i);
        return moves;
    }

    function detectTerminal(b) {
        const lines = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ];
        for (const [a,b1,c] of lines) {
            if (b[a] && b[a] === b[b1] && b[a] === b[c]) {
                return { terminal: true, winner: b[a], winningLine: [a,b1,c], draw:false };
            }
        }
        if (generateLegalMoves(b).length === 0) {
            return { terminal:true, winner:null, winningLine:null, draw:true };
        }
        return { terminal:false, winner:null, winningLine:null, draw:false };
    }

    function applyMove(currentBoard, index, symbol) {
        const nb = currentBoard.slice();
        nb[index] = symbol;
        return nb;
    }

    /* ===================== Classical Heuristic ===================== */
    function classicalHeuristic(boardState) {
        const t = detectTerminal(boardState);
        if (t.terminal) {
            if (t.winner === aiSymbol) return 1000;
            if (t.winner === playerSymbol) return -1000;
            return 0;
        }

        let score = 0;
        const lines = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ];

        for (const [a,b,c] of lines) {
            const vals = [boardState[a], boardState[b], boardState[c]];
            const aiCount = vals.filter(v => v === aiSymbol).length;
            const pCount  = vals.filter(v => v === playerSymbol).length;

            if (aiCount > 0 && pCount === 0) {
                if (aiCount === 2) score += 80;
                else score += 10;
            }
            else if (pCount > 0 && aiCount === 0) {
                if (pCount === 2) score -= 70;
                else score -= 8;
            }
        }

        if (boardState[4] === aiSymbol) score += 25;
        if (boardState[4] === playerSymbol) score -= 25;

        const corners = [0,2,6,8];
        for (const c of corners) {
            if (boardState[c] === aiSymbol) score += 6;
            if (boardState[c] === playerSymbol) score -= 6;
        }

        return score;
    }

    /* ===================== ML Eval Plug-in ===================== */
    function mlEvaluation(features) {
        // place your ML model output here
        return 0;
    }

    function extractFeaturesForML(boardState, aiSym) {
        const opponent = (aiSym === 'X') ? 'O' : 'X';
        const X_count = boardState.filter(c => c === 'X').length;
        const O_count = boardState.filter(c => c === 'O').length;

        const X_center = boardState[4] === 'X' ? 1 : 0;
        const O_center = boardState[4] === 'O' ? 1 : 0;

        const corners = [0,2,6,8];
        const X_corners = corners.filter(i => boardState[i] === 'X').length;
        const O_corners = corners.filter(i => boardState[i] === 'O').length;

        const lines = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ];
        let X_almost = 0;
        let O_almost = 0;
        for (const [a,b,c] of lines) {
            const vals = [boardState[a], boardState[b], boardState[c]];
            const x = vals.filter(v => v === 'X').length;
            const o = vals.filter(v => v === 'O').length;
            const empty = vals.filter(v => v === null).length;
            if (x === 2 && empty === 1) X_almost++;
            if (o === 2 && empty === 1) O_almost++;
        }

        if (aiSym === 'X') {
            return [X_count, O_count, X_almost, O_almost, X_center, X_corners];
        } else {
            return [O_count, X_count, O_almost, X_almost, O_center, O_corners];
        }
    }

    /* ===================== Alpha-Beta ===================== */
    function alphaBeta(nodeBoard, depth, alpha, beta, maximizingPlayer, evaluationFn) {
        const t = detectTerminal(nodeBoard);
        if (t.terminal || depth === 0) {
            return { score: evaluationFn(nodeBoard), move:null };
        }

        const legal = generateLegalMoves(nodeBoard);
        let bestMove = null;

        if (maximizingPlayer) {
            let value = -Infinity;
            for (const mv of legal) {
                const nb = applyMove(nodeBoard, mv, aiSymbol);
                const res = alphaBeta(nb, depth - 1, alpha, beta, false, evaluationFn);
                if (res.score > value) {
                    value = res.score;
                    bestMove = mv;
                }
                alpha = Math.max(alpha, value);
                if (alpha >= beta) break;
            }
            return { score:value, move:bestMove };
        }
        else {
            let value = Infinity;
            for (const mv of legal) {
                const nb = applyMove(nodeBoard, mv, playerSymbol);
                const res = alphaBeta(nb, depth - 1, alpha, beta, true, evaluationFn);
                if (res.score < value) {
                    value = res.score;
                    bestMove = mv;
                }
                beta = Math.min(beta, value);
                if (beta <= alpha) break;
            }
            return { score:value, move:bestMove };
        }
    }

    /* ===================== Difficulty (Pure Depth) ===================== */
    function getDifficultyParams(diff) {
        switch ((diff || 'normal').toLowerCase()) {
            case 'easy':   return { depth:2 };
            case 'normal': return { depth:4 };
            case 'hard':   return { depth:9 };
            default:       return { depth:4 };
        }
    }

    /* ===================== Rendering ===================== */
    function renderBoard() {
        boardEl.innerHTML = '';
        for (let i = 0; i < 9; i++) {
            const btn = document.createElement('button');
            btn.className = 'board-cell';
            btn.dataset.index = i;
            btn.innerText = board[i] || '';
            btn.disabled = gameOver || !!board[i];
            btn.addEventListener('click', () => onCellClick(i));
            boardEl.appendChild(btn);
        }

        const st = detectTerminal(board);
        if (st.terminal && st.winningLine) {
            for (const idx of st.winningLine) {
                const c = boardEl.querySelector(`button[data-index="${idx}"]`);
                if (c) c.classList.add('cell-win');
            }
        }
    }

    function setMessage(t) { gameMessage.textContent = t; }

    function showEvalList(evals) {
        evalList.innerHTML = '';
        const sorted = evals.slice().sort((a,b) => b.score - a.score);
        for (const e of sorted) {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = `Move ${e.move+1} → score ${e.score}`;
            evalList.appendChild(li);
        }
    }

    /* ===================== Player Move ===================== */
    function onCellClick(i) {
        if (gameOver || board[i]) return;

        board = applyMove(board, i, playerSymbol);
        renderBoard();

        const st = detectTerminal(board);
        if (st.terminal) return finishGame(st);

        setTimeout(() => aiMove(), 120);
    }

    /* ===================== AI Move ===================== */
    function aiMove() {
        if (gameOver) return;

        const params = getDifficultyParams(settings.difficulty);
        const depth = params.depth;

        const evaluationFn = (settings.evaluation === 'ml')
            ? (b => mlEvaluation(extractFeaturesForML(b, aiSymbol)))
            : classicalHeuristic;

        const legal = generateLegalMoves(board);
        if (legal.length === 0) return;

        const evals = [];
        for (const mv of legal) {
            const nb = applyMove(board, mv, aiSymbol);
            const res = alphaBeta(nb, depth-1, -Infinity, Infinity, false, evaluationFn);
            evals.push({ move:mv, score:res.score });
        }
        showEvalList(evals);

        const best = alphaBeta(board, depth, -Infinity, Infinity, true, evaluationFn);
        const chosen = best.move ?? legal[0];

        aiDecision.textContent = `AI plays: ${chosen+1}`;

        board = applyMove(board, chosen, aiSymbol);
        renderBoard();

        const st = detectTerminal(board);
        if (st.terminal) return finishGame(st);

        setMessage('Your turn.');
    }

    /* ===================== Finish Game ===================== */
    function finishGame(st) {
        gameOver = true;
        if (st.winner === playerSymbol) setMessage('You win!');
        else if (st.winner === aiSymbol) setMessage('AI wins!');
        else setMessage('Draw.');
    }

    /* ===================== Buttons ===================== */
    backBtn.addEventListener('click', () => window.location.href = 'index.html');

    resetBtn.addEventListener('click', () => {
        board = Array(9).fill(null);
        gameOver = false;
        aiDecision.textContent = '—';
        evalList.innerHTML = '';
        renderBoard();

        if (playerSymbol === 'X') setMessage('Game started. Your move.');
        else {
            setMessage('AI starts.');
            setTimeout(() => aiMove(), 200);
        }
    });

    /* ===================== Init ===================== */
    board = Array(9).fill(null);
    gameOver = false;
    renderBoard();

    if (playerSymbol === 'X') setMessage('Game started. Your move.');
    else {
        setMessage('AI starts.');
        setTimeout(() => aiMove(), 200);
    }

    window.__ttt = { board, alphaBeta, classicalHeuristic, mlEvaluation, extractFeaturesForML };
});
