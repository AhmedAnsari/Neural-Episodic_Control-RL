package.path = package.path .. ';lua/?/init.lua'
g_mazebase = require('mazebase')

g_opts = {}
g_opts.games_config_path = 'lua/mazebase/config/game_config.lua'
-- g_opts.games_config_path = 'lua/mazebase/config/singlegoal.lua'
--g_opts.game = 'MultiGoals'
-- g_opts.game = 'LightKey'
g_opts.game = 'SingleGoal'

g_mazebase.init_vocab()  --modify init_vocab for new objects
g_mazebase.init_game() -- init.lua >> call init_game >> g_opts from config file

g = g_mazebase.new_game() -- init.lua >> new_game() >> init_game >> gfactory

-- print(g_opts)

g_disp = require'display'
nactions = #g.agent.action_names
 
print(g.agent.action_names)

for t = 1, 240 do
	g_disp.image(g.map:to_image(), {win='fixed'})
	-- local s = g:to_sentence()
	-- print(s)
	g:act(torch.random(nactions))
	g:act(6)
	g:act(7)
	g:act(8)
	g:act(9)
	g:update()
	if g:is_active() == false then
		break
	end
	os.execute('sleep 0.2')
end
