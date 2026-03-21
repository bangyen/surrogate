/**
 * Chess AI Dashboard - Main application logic and API integration.
 */

class ChessApp {
    constructor() {
        this.board = new ChessBoard('chess-board');
        this.board.onMove = (move) => this.handlePlayerMove(move);
        this.moveCount = 0;

        this.initializeNavigation();
        this.initializeEventListeners();
        this.initializeGame();
    }

    initializeNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const views = document.querySelectorAll('.view-container');

        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const viewName = item.getAttribute('data-view');

                // Update active states
                navItems.forEach(n => n.classList.remove('active'));
                item.classList.add('active');

                views.forEach(v => v.classList.add('hidden'));
                document.getElementById(`${viewName}-view`).classList.remove('hidden');

                // Update page title
                const titles = {
                    'play': 'Play Chess',
                    'features': 'Position Features'
                };
                document.querySelector('.page-title').textContent = titles[viewName] || 'Chess AI';
            });
        });
    }

    initializeEventListeners() {
        document.getElementById('btn-new-game').addEventListener('click', () => {
            this.initializeGame();
        });

        const undoBtn = document.getElementById('btn-undo-move');
        if (undoBtn) {
            undoBtn.addEventListener('click', () => {
                this.undoMove();
            });
        }

        const switchBtn = document.getElementById('btn-switch-sides');
        if (switchBtn) {
            switchBtn.addEventListener('click', () => {
                this.switchSides();
            });
        }

        document.getElementById('btn-analyze').addEventListener('click', () => {
            this.analyzePosition();
        });
    }

    async initializeGame() {
        try {
            const response = await fetch('/api/game/new', {
                method: 'POST'
            });

            const data = await response.json();
            this.board.setPosition(data.fen, null);
            this.board.setLegalMoves(data.legal_moves);
            this.board.setOrientation('white');
            this.moveCount = 0;

            this.updateGameState();
            this.clearExplanation();

            // Start polling for engine status
            this.pollEngineStatus();
        } catch (error) {
            console.error('Failed to initialize game:', error);
            this.updateEngineStatus('Error', false);
        }
    }

    async pollEngineStatus() {
        const checkStatus = async () => {
            try {
                const response = await fetch('/api/engine/status');
                const data = await response.json();

                if (data.model_ready) {
                    this.updateEngineStatus('Ready', true);
                    return true; // Stop polling
                } else if (data.error) {
                    this.updateEngineStatus('Engine Error', false);
                    return true; // Stop polling
                } else {
                    this.updateEngineStatus('Training Model...', true, true);
                    return false; // Continue polling
                }
            } catch (error) {
                this.updateEngineStatus('Status Error', false);
                return true;
            }
        };

        const ready = await checkStatus();
        if (!ready) {
            const interval = setInterval(async () => {
                if (await checkStatus()) {
                    clearInterval(interval);
                }
            }, 2000);
        }
    }

    async handlePlayerMove(move, isEngineMove = false, explanation = null) {
        try {
            const response = await fetch('/api/game/move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ move })
            });

            const data = await response.json();

            if (data.success) {
                this.board.setPosition(data.fen, move);
                this.board.setLegalMoves(data.legal_moves);
                this.moveCount++;

                // Get position features and display
                const features = await this.getPositionFeatures();
                if (features) {
                    // Display features for manual moves too
                    if (!isEngineMove) {
                        this.displayManualMoveInfo(features);
                    } else if (explanation) {
                        this.displayExplanation(explanation, features);
                    }
                }

                this.updateGameState();

                if (data.is_game_over) {
                    this.handleGameOver();
                } else if (!isEngineMove) {
                    // Automatically trigger engine move after player move
                    setTimeout(() => this.requestEngineMove(), 500);
                }
            }
        } catch (error) {
            console.error('Failed to make move:', error);
        }
    }

    async requestEngineMove() {
        const undoBtn = document.getElementById('btn-undo-move');
        const switchBtn = document.getElementById('btn-switch-sides');
        
        if (switchBtn) {
            switchBtn.classList.add('loading');
            switchBtn.disabled = true;
        }
        if (undoBtn) {
            undoBtn.disabled = true;
        }

        this.updateEngineStatus('Thinking...', true);

        try {
            const response = await fetch('/api/engine/move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ depth: 15 })
            });

            const data = await response.json();

            if (data.move) {
                await this.handlePlayerMove(data.move, true, data.explanation);
                this.updateEngineStatus('Ready', true);
            }
        } catch (error) {
            console.error('Failed to get engine move:', error);
            this.updateEngineStatus('Error', false);
        } finally {
            const undoBtn = document.getElementById('btn-undo-move');
            const switchBtn = document.getElementById('btn-switch-sides');
            
            if (switchBtn) {
                switchBtn.classList.remove('loading');
                switchBtn.disabled = false;
            }
            if (undoBtn) {
                undoBtn.disabled = false;
            }
        }
    }

    async undoMove() {
        try {
            // Undo full turn (Player + Engine move) in one call
            const response = await fetch('/api/game/undo', { method: 'POST' });
            
            if (response.ok) {
                const data = await response.json();
                this.board.setPosition(data.fen, null);
                this.board.setLegalMoves(data.legal_moves);
                this.moveCount = Math.max(0, this.moveCount - 2);
                this.updateGameState();
                this.clearExplanation();
                
                // Re-analyze position
                const features = await this.getPositionFeatures();
                if (features) {
                    this.displayManualMoveInfo(features);
                }
            } else {
                console.warn('Could not undo further.');
            }
        } catch (error) {
            console.error('Failed to undo move:', error);
        }
    }

    async switchSides() {
        // Toggle board orientation
        const newOrientation = this.board.orientation === 'white' ? 'black' : 'white';
        this.board.setOrientation(newOrientation);
        
        // Trigger automated engine response for the new side
        await this.requestEngineMove();
    }

    async analyzePosition() {
        try {
            const response = await fetch('/api/analysis/features', {
                method: 'GET'
            });

            const data = await response.json();
            this.displayFeatures(data.features);
        } catch (error) {
            console.error('Failed to analyze position:', error);
        }
    }

    async getPositionFeatures() {
        try {
            const response = await fetch('/api/analysis/features', {
                method: 'GET'
            });
            const data = await response.json();
            return data.features;
        } catch (error) {
            console.error('Failed to get position features:', error);
            return null;
        }
    }

    async updateGameState() {
        try {
            const response = await fetch('/api/game/state');
            const data = await response.json();

            // Update board label
            const boardLabel = document.getElementById('board-label');
            if (boardLabel) {
                boardLabel.textContent = data.turn === 'white' ? 'White to move' : 'Black to move';
            }

            // Update engine status based on game state
            if (data.is_game_over) {
                this.handleGameOver();
            }
        } catch (error) {
            console.error('Failed to update game state:', error);
        }
    }

    displayManualMoveInfo(features) {
        const container = document.getElementById('explanation-content');
        const keyFeaturesContainer = document.getElementById('key-features');

        // Clear explanation for manual moves
        container.innerHTML = '<p class="text-muted">Request an engine move to see analysis.</p>';

        // Display top 3 features
        this.displayTopFeatures(features, keyFeaturesContainer);
    }

    displayExplanation(explanation, features) {
        const container = document.getElementById('explanation-content');
        const keyFeaturesContainer = document.getElementById('key-features');

        if (!explanation) {
            container.innerHTML = '<p class="text-muted">No explanation available.</p>';
            return;
        }

        // Parse explanation into paragraphs and lists
        // Normalize ' · ' into newlines for backward compatibility
        const normalizedExplanation = explanation.replace(/ · /g, '\n- ');
        const lines = normalizedExplanation.split('\n');
        
        let html = '';
        let listItems = [];

        lines.forEach(line => {
            const trimmedLine = line.trim();
            if (!trimmedLine) return;

            if (trimmedLine.startsWith('-') || trimmedLine.startsWith('•')) {
                // Remove the bullet and trim
                const content = trimmedLine.replace(/^[-•]\s*/, '').trim();
                listItems.push(this.processExplanationText(content));
            } else {
                // If we were building a list, close it out
                if (listItems.length > 0) {
                    html += `<ul>${listItems.map(item => `<li>${item}</li>`).join('')}</ul>`;
                    listItems = [];
                }
                html += `<p>${this.processExplanationText(trimmedLine)}</p>`;
            }
        });

        // Close final list if exists
        if (listItems.length > 0) {
            html += `<ul>${listItems.map(item => `<li>${item}</li>`).join('')}</ul>`;
        }

        container.innerHTML = html;

        // Display top 3 features
        this.displayTopFeatures(features, keyFeaturesContainer);
    }

    displayTopFeatures(features, container) {
        if (!container) return;

        // features is now an array of [raw_name, label, value]
        const topFeatures = features.slice(0, 3);

        if (topFeatures.length > 0) {
            let featuresHtml = '';
            topFeatures.forEach(([_, label, value]) => {
                const formattedName = label;
                const formattedValue = value.toFixed(2) + ' cp';
                featuresHtml += `
                    <div class="feature-item">
                        <span class="feature-name">${formattedName}</span>
                        <span class="feature-value">${formattedValue}</span>
                    </div>
                `;
            });
            container.innerHTML = featuresHtml;
        }
    }

    processExplanationText(text) {
        // Add newline before (XX cp) to avoid wrapping
        let processed = text.replace(/ \((.*? cp)\)/g, '<br><span class="evaluation">($1)</span>');
        // Basic bold formatting: **text** -> <strong>text</strong>
        return processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }

    clearExplanation() {
        const container = document.getElementById('explanation-content');
        const keyFeaturesContainer = document.getElementById('key-features');

        container.innerHTML = '<p class="text-muted">Request an engine move to see analysis.</p>';
        if (keyFeaturesContainer) {
            keyFeaturesContainer.innerHTML = '<p class="text-muted">Key features will appear here after each move.</p>';
        }
    }

    formatFeatureName(name) {
        // Fallback for cases where label isn't provided or we only have raw name
        let formatted = name
            .replace(/_us$/, ' (White)')
            .replace(/_them$/, ' (Black)');

        // Convert snake_case to Title Case
        formatted = formatted
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');

        return formatted;
    }

    displayFeatures(features) {
        const container = document.getElementById('features-table');
        if (!container) return;

        // features is [raw_name, label, value]
        const topFeatures = features.filter(([_, __, value]) => Math.abs(value) >= 10.0);
        const otherFeatures = features.filter(([_, __, value]) => Math.abs(value) < 10.0);

        let html = '';
        
        // Render top features
        topFeatures.forEach(([_, label, value]) => {
            const formattedName = label;
            const formattedValue = value.toFixed(2) + ' cp';
            html += `
                <div class="feature-row">
                    <span class="feature-name">${formattedName}</span>
                    <span class="feature-value ${value >= 0 ? 'positive' : 'negative'}">${formattedValue}</span>
                </div>
            `;
        });

        // Render other features behind an accordion if any exist
        if (otherFeatures.length > 0) {
            html += `
                <div class="accordion-section">
                    <button class="accordion-header" id="toggle-others">
                        <span>Less Significant Features (${otherFeatures.length})</span>
                        <span class="chevron">▼</span>
                    </button>
                    <div class="accordion-content hidden" id="others-container">
            `;
            
            otherFeatures.forEach(([_, label, value]) => {
                const formattedName = label;
                const formattedValue = value.toFixed(2) + ' cp';
                html += `
                    <div class="feature-row sub-feature">
                        <span class="feature-name">${formattedName}</span>
                        <span class="feature-value ${value >= 0 ? 'positive' : 'negative'}">${formattedValue}</span>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;

        // Add event listener for the accordion
        const toggleBtn = document.getElementById('toggle-others');
        if (toggleBtn) {
            toggleBtn.onclick = () => {
                const othersContainer = document.getElementById('others-container');
                const chevron = toggleBtn.querySelector('.chevron');
                const isHidden = othersContainer.classList.contains('hidden');
                
                if (isHidden) {
                    othersContainer.classList.remove('hidden');
                    chevron.textContent = '▲';
                } else {
                    othersContainer.classList.add('hidden');
                    chevron.textContent = '▼';
                }
            };
        }
    }

    updateEngineStatus(status, isActive, isWaiting = false) {
        const statusText = document.getElementById('engine-status-text');
        const statusDot = document.getElementById('engine-status-dot');

        if (statusText) {
            statusText.textContent = status;
        }

        if (statusDot) {
            statusDot.className = 'status-dot'; // Reset classes
            if (isWaiting) {
                statusDot.classList.add('waiting');
            } else if (!isActive) {
                statusDot.classList.add('error');
            }
        }
    }

    handleGameOver() {
        this.updateEngineStatus('Game Over', false);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ChessApp();
});
