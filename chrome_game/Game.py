import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from gym_cartpole.RL_brain import DeepQNetwork

game_url = "http://www.baidu.com"
chrome_driver_path = "/Users/cc/cc/DinoRunTutorial/chromedriver"
loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
q_value_file_path = "./objects/q_values.csv"
scores_file_path = "./objects/scores_df.csv"
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"


class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get(game_url)
        self._driver.execute_script(init_script)

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")

        score = ''.join(
            score_array)  # the javascript object is of type array with score in the formate[1,0,0] which is 100.

        try:
            return int(score)
        except:
            return 0

    # 跳跃次数
    def jumpCount(self):
        return self._driver.execute_script("return Runner.instance_.tRex.jumpCount")

    def get_state(self):
        obstacle = self._driver.execute_script("return Runner.instance_.horizon.obstacles[0]")
        tRex = self._driver.execute_script("return Runner.instance_.tRex")

        if obstacle is None:
            s = np.array([])
        else:
            # if (tRexBox.x < obstacleBoxX + obstacleBox.width & &
            #         tRexBox.x + tRexBox.width > obstacleBoxX & &
            #         tRexBox.y < obstacleBox.y + obstacleBox.height & &
            #         tRexBox.height + tRexBox.y > obstacleBox.y) {
            # crashed = true;
            # }
            tRexX, tRexY, tRexW, tRexH = tRex['xPos'] + 1, tRex['yPos'] + 1, tRex['config']['WIDTH'] - 2, \
                                         tRex['config'][
                                             'HEIGHT'] - 2

            obstacleX, obstacleY, obstacleW, obstacleH = obstacle['xPos'] + 1, obstacle['yPos'] + 1, \
                                                         obstacle['typeConfig'][
                                                             'width'] * obstacle['size'] - 2, obstacle['typeConfig'][
                                                             'height'] - 2

            s = (obstacleX + obstacleW) - tRexX, obstacleX - (tRexX + tRexW), (
                    obstacleY + obstacleH) - tRexY, tRexH + tRexY - obstacleY, self.get_score() // 100

        return np.array(s)

    def get_jumping(self):
        return self._driver.execute_script("return Runner.instance_.tRex.jumping")

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()


RL = DeepQNetwork(n_actions=2,
                  n_features=5,
                  learning_rate=0.01, e_greedy=0.95,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.0001, )

total_steps = 0
if __name__ == '__main__':
    env = Game()
    for i in range(1000):
        env.restart()
        env.press_up()
        state = env.get_state()
        reward_conter = 0

        while True:
            if len(state) == 0 or env.get_state()[0] > 150 + env.get_state()[-1:][0] * 10:
                state = env.get_state()
                continue

            if not env.get_crashed():  # game over
                if env.get_jumping():
                    continue

            action = RL.choose_action(state)

            if action == 1:
                env.press_up()

            if env.get_crashed():  # game over
                state_ = env.get_state()
                RL.store_transition(state, action, -1, state_)

                total_steps = total_steps + 1
                print("reward_conter", reward_conter, "total_steps", total_steps, "epsilon", RL.epsilon)

                break;
            else:
                state_ = env.get_state()
                reward = 0
                if action == 1:
                    reward = 1 / np.sqrt(np.sum(np.square(state - state_)))

                RL.store_transition(state, action, reward, state_)
                state = state_
                total_steps = total_steps + 1
                reward_conter += reward

            if total_steps > 100:
                RL.learn()

env._driver.close()
