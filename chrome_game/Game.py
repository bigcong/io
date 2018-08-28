from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

game_url = "file:///Users/cc/cc/t-rex-runner/index.html"
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
        self._driver = webdriver.Chrome(executable_path=chrome_driver_path, chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get(game_url)
        self._driver.execute_script("Runner.config.ACCELERATION=0")
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

    def get_state(self):
        s = self._driver.execute_script("return Runner.instance_.horizon.obstacles[0]")
        s_ = self._driver.execute_script("return Runner.instance_.tRex")

        return s, s_

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()


if __name__ == '__main__':
    g = Game()
    for i in range(200):
        g.restart()
        while True:
            s,s_=g.get_state()
            print(s)
            if g.get_crashed():
                break;
            else:
                g.press_up()
            print(g.get_score())

g.end()
