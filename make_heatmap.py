import random
import time
import io
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import numpy as np 
#迷路の生成用クラス
class MazeEnvironment:
    def __init__(self, rows=25, cols=25, wall_rate=0.2):
        self.rows = rows
        self.cols = cols
        self.start_pos = (0, 0)
        self.goal_pos = (rows - 1, cols - 1)
        self.agent_pos = self.start_pos
        self.grid_map = [[0 for _ in range(cols)] for _ in range(rows)]
        
        #seed=5でゴール到達可能な迷路となることを検証済み
        random.seed(5) 
        
        for r in range(rows):
            for c in range(cols):
                if (r, c) == self.start_pos or (r, c) == self.goal_pos:
                    continue
                if random.random() < wall_rate:
                    self.grid_map[r][c] = 1
    
    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        row, col = self.agent_pos
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = moves[action]
        next_row, next_col = row + dr, col + dc
        
        reward = -1
        done = False
        
        if (next_row < 0 or next_row >= self.rows or
            next_col < 0 or next_col >= self.cols or
            self.grid_map[next_row][next_col] == 1):
            next_state = (row, col)
        else:
            self.agent_pos = (next_row, next_col)
            next_state = self.agent_pos
            
        if next_state == self.goal_pos:
            reward = 200
            done = True
            
        return next_state, reward, done

#強化学習(QLearning)エージェントのクラス
#動作原理はε-greedy。一定確率εで探索を行い、それ以外で搾取を行う
class QLearningAgent:
    def __init__(self, rows, cols):
        self.q_table = {}
        for r in range(rows):
            for c in range(cols):
                self.q_table[(r, c)] = [0.0] * 4 
        self.epsilon = 0.1
        #alpha:学習率 ある回にうまくいった経験をどれくらい信じるか、という変数。小さいほど慎重になる。
        self.alpha = 0.1
        #gamma:時間割引率 n手先の行動の影響力がgamma**n倍になる。大きいほど未来を見通せるようになる。
        self.gamma = 0.9

    def get_action(self, state, mode='train'):
        if mode == 'train' and random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            #trainモードでないときはε=0の挙動を示す(今回は使っていない)
            q_values = self.q_table[state]
            max_q = max(q_values)
            if min(q_values) == max_q:
                return random.randint(0, 3)
            actions_with_max_q = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(actions_with_max_q)

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q

def create_snapshot_images(env, agent, end_episode, interval_visit_counts, start_episode):
    """
    その時点での「テスト走行軌跡」と「特定区間のヒートマップ」の両方を保存する
    """
    interval_label = f"Ep {start_episode}-{end_episode}"
    print(f"--- 画像生成中: {interval_label} ---")
    
    # 1. テスト走行の軌跡画像 (現在にエージェントの実力確認用)
    state = env.reset()
    done = False
    path_history = [state]
    steps = 0
    while not done and steps < 300:
        action = agent.get_action(state, mode='test')
        next_state, _, done = env.step(action)
        state = next_state
        steps += 1
        path_history.append(state)
        if done: break
            
    plt.figure(figsize=(8, 8))
    plt.imshow(env.grid_map, cmap='binary', interpolation='nearest')
    path_rows = [p[0] for p in path_history]
    path_cols = [p[1] for p in path_history]
    plt.plot(path_cols, path_rows, color='blue', alpha=0.5, linewidth=2)
    plt.plot(path_cols[-1], path_rows[-1], marker='o', color='red', markersize=8)
    status = "GOAL!" if done else "FAILED"
    
    plt.title(f"Test Run: After {start_episode} Episodes ({status})")
    
    # グリッド線
    ax = plt.gca()
    ax.set_xticks([x - 0.5 for x in range(1, env.cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, env.rows)], minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    plt.axis('on')

    # ファイル名も区間がわかるように変更
    path_filename = f"path_ep{start_episode}.png"
    plt.savefig(path_filename)
    plt.close()

    # 2. 指定区間(n ~ n+100)のヒートマップ画像
    plt.figure(figsize=(8, 8))
    plt.imshow(env.grid_map, cmap='binary', interpolation='nearest', alpha=0.3)
    
    visit_array = np.array(interval_visit_counts)
    masked_visits = np.ma.masked_where(visit_array == 0, visit_array)
    
    plt.imshow(masked_visits, cmap='hot_r', interpolation='nearest', alpha=0.8)
    
    # 罫線を追加
    ax = plt.gca()
    ax.set_xticks([x - 0.5 for x in range(1, env.cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, env.rows)], minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.5)
    ax.tick_params(which="minor", size=0)
    
    max_visit = np.max(visit_array)
    plt.title(f"Heatmap: {interval_label} (Window: 100 eps)\n(Max Visits: {max_visit})")
    plt.axis('on')
    
    heatmap_filename = f"heatmap_interval_{start_episode}-{end_episode}.png"
    plt.savefig(heatmap_filename)
    plt.close()
    
    print(f"保存: {path_filename}, {heatmap_filename}")

def main():
    ROWS = 25
    COLS = 25
    
    env = MazeEnvironment(rows=ROWS, cols=COLS, wall_rate=0.4)
    agent = QLearningAgent(env.rows, env.cols)
    
    # ここで指定したエピソード数から、+100回分だけデータを詳しく取る
    checkpoints = [200, 2000, 10000]
    monitoring_window = 100  # n回目〜n+100回目を見る
    
    # 学習終了エピソードを延長する
    total_episodes = max(checkpoints) + monitoring_window
    
    # 計測用データ保持用
    # active_monitoring: 現在計測中かどうかを管理
    # { 'start_ep': 200, 'counts': [[...]] } のような辞書を持つ。なければNone
    active_monitoring = None
    
    print(f"学習開始（最大{total_episodes}エピソードまで延長）")
    
    # rangeは1から開始
    for episode in range(1, total_episodes + 1):
        
        # --- チェックポイント到達時の計測開始処理 ---
        if episode in checkpoints:
            if active_monitoring is not None:
                # すでに計測中の場合（間隔が狭い場合など）は前のを強制終了して保存も可能だが
                # 今回の設定では被らない前提で上書き、または無視
                pass
            
            print(f"-> Ep {episode} 到達。ここから{monitoring_window}エピソード分の行動を記録します。")
            active_monitoring = {
                'start_ep': episode,
                'end_ep': episode + monitoring_window,
                'counts': [[0 for _ in range(COLS)] for _ in range(ROWS)]
            }

        # --- エピソード実行 ---
        state = env.reset()
        done = False
        steps = 0
        
        # スタート地点カウント（計測中なら）
        if active_monitoring:
            active_monitoring['counts'][state[0]][state[1]] += 1
        
        while not done and steps < ROWS * COLS * 2:
            action = agent.get_action(state, mode='train')
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            steps += 1
            
            # 移動先カウント（計測中なら）
            if active_monitoring:
                active_monitoring['counts'][state[0]][state[1]] += 1
        
        # --- 計測終了判定と保存処理 ---
        if active_monitoring:
            if episode == active_monitoring['end_ep']:
                # 指定期間（100エピソード）終了。画像を生成して計測モードを解除
                create_snapshot_images(
                    env, 
                    agent, 
                    episode, # end_episode
                    active_monitoring['counts'], 
                    active_monitoring['start_ep']
                )
                active_monitoring = None

    print("\n完了しました。")

if __name__ == "__main__":
    main()