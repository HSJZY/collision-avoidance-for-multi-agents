#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:22:24 2018

@author: jiang
"""

import math
from collections import namedtuple
import pyglet
import random
import numpy as np

class Agent(object):
    def __init__(self, px, py, pgx, pgy, radius,sensor_radius=50):
        self.px = px
        self.py = py
        self.vx = 0
        self.vy = 0
        self.radius = radius
        self.pgx = pgx
        self.pgy = pgy
        self.done = False
        self.sensor_radius=sensor_radius

    def update_state(self, action, time):
        action.vx=action.vx*20
        action.vy=action.vy*20
        self.vx=action.vx
        self.vy=action.vy
        self.px, self.py = self.compute_position(time=time, action=action)

    def reset(self):
        pass

    def get_full_state(self):
        return self.px, self.py, self.vx, self.vy, self.radius, self.pgx, self.pgy

    def get_sensor_radius(self):
        return self.sensor_radius
    
    def get_agent_radius(self):
        return self.radius
    
    def get_observable_state(self):
        return self.px, self.py, self.pgx, self.pgy
    
    def get_position(self):
        return (self.px,self.py)

    def compute_position(self, time, action=None):
        if action is None:
            # assume the agent travels in original speed
            x = self.px + time * self.vx
            y = self.py + time * self.vy
        else:
            x=self.px+time*action.vx
            y=self.py+time*action.vy
        return x, y
class Obstacle():
    def __init__(self,px,py,radius):
        self.px=px
        self.py=py
        self.radius=radius
    def get_observable_state(self):
        return self.px,self.py,self.radius
    
class ENV(object):
    action_bound = [-1, 1]
    action_dim = 2
    point_l = 5
    num_sensor=16
    state_dim = num_sensor+3
    viewer = None
    
    def __init__(self,num_agents,num_obstacles,agent_radius,viewer_xy):
        self.num_agents=num_agents
        self.agents=[None]*num_agents
        self.x_min=0
        self.y_min=0
        self.x_max=viewer_xy[0]
        self.y_max=viewer_xy[1]
        self.dt=0.1
        self.viewer_xy = viewer_xy
        self.obstacles=[None]*num_obstacles
        self.agent_radius=agent_radius
        self.proportion=50
        self.grab_counter=[0]*num_agents
        self.get_point=[False]*num_agents
        
        
        
    def step(self,actions):
        r=[]
        s=[]
        dis_to_goal=[]
        for i in range(self.num_agents):
            action_i=actions[i]
            r_i=self.compute_reward(i,action_i)
            s_i,dis_xy_i=self._get_state(i)
            r.append(r_i)
            s.append(s_i)
            dis_to_goal.append(dis_xy_i)
        return s,r,dis_to_goal
    
    def compute_reward(self,agent_id,action):
        def _r_func( abs_distance,idx):
            t = 50
            r = -abs_distance/200
            if abs_distance < self.point_l and (not self.get_point[idx]):
                r += 1.
                self.grab_counter[idx] += 1
                if self.grab_counter[idx] > t:
                    r += 10.
                    self.get_point[idx] = True
            elif abs_distance > self.point_l:
                self.grab_counter[idx] = 0
                self.get_point[idx] = False
            return r
        
        agent=self.agents[agent_id]
        if agent==None:
            return 0
        assert len(action)==2
        _action=namedtuple('vx','vy')
        _action.vx=action[0]
        _action.vy=action[1]
        agent.update_state(_action,self.dt)
        agent_px,agent_py,agent_pgx,agent_pgy=agent.get_observable_state()
        reward=0
        if(self._check_collision(agent_id=agent_id)):
            reward=-10.
        else:
            distance_to_goal=math.sqrt(pow(agent_pgx-agent_px,2)+pow(agent_pgy-agent_py,2))
            #print("distance_to_goal:",distance_to_goal)
            reward=_r_func(distance_to_goal,agent_id)
        return reward
        

    def for_testing(self):
        return self.agents,self.obstacles
        
    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.agents, self.obstacles)
        else:
            self.viewer.update(self.agents)
            self.viewer.render()
    
    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)
    
    def _reset_agents(self):
        self.agents=[None]*self.num_agents
        self.get_point=[False]*self.num_agents
        for i in range(len(self.agents)):
            is_collision=True
            while is_collision:
                _px=random.randint(self.x_min+self.agent_radius,self.x_max-self.agent_radius)
                _py=random.randint(self.y_min+self.agent_radius,self.y_max-self.agent_radius)
                _agent_i=Agent(_px,_py,-1,-1,self.agent_radius)
                self.agents[i]=_agent_i
                is_collision=self._check_collision(agent_id=i)
            is_collision=True
            px,py=self.agents[i].get_position()
            while is_collision:
                _pgx=random.randint(self.x_min+self.agent_radius,self.x_max-self.agent_radius)
                _pgy=random.randint(self.y_min+self.agent_radius,self.y_max-self.agent_radius)
                _agent_i=Agent(_pgx,_pgy,-1,-1,self.agent_radius)
                self.agents[i]=_agent_i
                is_collision=self._check_collision(agent_id=i)
            pgx,pgy=self.agents[i].get_position()
            agent_i=Agent(px,py,pgx,pgy,self.agent_radius)
            self.agents[i]=agent_i
            
            
    def _reset_obstacles(self,radius_bound):
        assert len(radius_bound)==2
        self.obstacles=[None]*len(self.obstacles)
        for i in range(len(self.obstacles)):
            is_collision=True
            while is_collision:
                _randius=random.randint(radius_bound[0],radius_bound[1])
                _px=random.randint(self.x_min+_randius,self.x_max-_randius)
                _py=random.randint(self.y_min+_randius,self.y_max-_randius)
                _obstacle=Obstacle(_px,_py,_randius)
                self.obstacles[i]=_obstacle
                is_collision=self._check_collision(obstacle_id=i)
    
    def reset(self):
        obstacle_radius_bound=[10,50]
        self.agents=[None]*len(self.agents)
        self._reset_obstacles(obstacle_radius_bound)
        self._reset_agents()
                
    def _check_collision(self,agent_id=None,obstacle_id=None):
        #用于检测机器人于其他机器人是否碰撞
        for idx in range(len(self.agents)):
            if agent_id==None:
                break
            if idx==agent_id or self.agents[idx]==None:
                continue
            if math.sqrt(pow(self.agents[idx].get_position()[0]-self.agents[agent_id].get_position()[0],2)+pow(self.agents[idx].get_position()[1]-self.agents[agent_id].get_position()[1],2))<2*self.agent_radius:
                return True
            
        #用于检测机器人于障碍物是否碰撞
        for idx in range(len(self.obstacles)):
            if agent_id==None:
                break
            if self.obstacles[idx]==None:
                continue
            _px,_py,_radius=self.obstacles[idx].get_observable_state()
            _agent_px,_agent_py=self.agents[agent_id].get_position()
            
            if math.sqrt((_agent_px-_px)**2+(_agent_py-_py)**2)<_radius+self.agent_radius:
                return True
        #用于检测机器人和边界是否碰撞
        if agent_id!=None:
            _agent_px,_agent_py=self.agents[agent_id].get_position()
            if _agent_px<self.agent_radius or _agent_px>self.x_max-self.agent_radius or _agent_py<self.agent_radius or _agent_py >self.y_max- self.agent_radius:
                return True
            
        #用于检测新生成的障碍物是否与其他障碍物发生碰撞
        for idx in range(len(self.obstacles)):
            if obstacle_id==None:
                break
            if idx==obstacle_id or self.obstacles[idx]==None:
                continue
            _px,_py,_radius=self.obstacles[idx].get_observable_state()
            _self_px,_self_py,_self_radius=self.obstacles[obstacle_id].get_observable_state()
            if math.sqrt(pow(_px-_self_px,2)+pow(_py-_self_py,2))<_radius+_self_radius:
                return True
        #用于检测新生成的障碍物是否与已有的机器人发生碰撞
        for idx in range(len(self.agents)):
            if(obstacle_id==None):
                break
            if self.agents[idx]==None:
                continue
            _self_px,_self_py,_self_radius=self.obstacles[obstacle_id].get_observable_state()
            _agent_px,_agent_py=self.agents[idx].get_position()
            if math.sqrt(pow(_self_px-_agent_px,2),pow(_self_py-_agent_py,2))<_self_radius+self.agent_radius:
                return True
        return False
    
    def sample_action(self):
        actions=[]
        for i in range(self.num_agents):
            actions.append(np.random.uniform(*self.action_bound, size=self.action_dim))
        return np.array(actions)
    
    
    def _calc_sensor_info(self,idx):
        def update_s(s,relative_px,relative_py,num_sensor,ob_radius,sensor_radius):
            assert len(s)==num_sensor
            dist2_cur_agent=math.sqrt(pow(relative_px,2)+pow(relative_py,2))
            if dist2_cur_agent-ob_radius>sensor_radius:
                return
            if dist2_cur_agent<ob_radius:
                for i in range(len(s)):
                    s[i]=1.
                return
            center_angle_rad=math.atan2(relative_py,relative_px)
            split_angle=abs(math.atan2(ob_radius,math.sqrt(pow(dist2_cur_agent,2)-pow(ob_radius,2))))
            start_angle=center_angle_rad-split_angle
            end_angle=center_angle_rad+split_angle
            if start_angle<0:
                start_angle+=2*math.pi
            if end_angle<0:
                end_angle+=2*math.pi
            if center_angle_rad<0:
                center_angle_rad+=2*math.pi
            each_angle=2*math.pi/num_sensor
            start_i=math.floor(start_angle/each_angle)
            end_i=math.ceil(end_angle/each_angle)
            center_i=center_angle_rad/each_angle
            for j in range(start_i,end_i):
                if center_i>j and center_i<j+1:
                    s[j]=max((sensor_radius-(dist2_cur_agent-ob_radius))/sensor_radius,s[j])
                    continue
                elif center_i>j:
                    line_j=j+1
                    length2_obstacle=dist2_cur_agent*math.cos(center_angle_rad-each_angle*line_j)-math.sqrt(ob_radius**2-(dist2_cur_agent*math.sin(center_angle_rad-each_angle*line_j))**2)
                    s[j]=max((sensor_radius-length2_obstacle)/sensor_radius,s[j])
                    #print("length2_obstacle:",length2_obstacle,"center_angle_rad-each_angle*line_j:",center_angle_rad-each_angle*line_j)
                elif center_i<j:
                    line_j=j
                    length2_obstacle=dist2_cur_agent*math.cos(center_angle_rad-each_angle*line_j)-math.sqrt(ob_radius**2-(dist2_cur_agent*math.sin(center_angle_rad-each_angle*line_j))**2)
                    s[j]=max((sensor_radius-length2_obstacle)/sensor_radius,s[j])                
        
        agent=self.agents[idx]
        _self_px,_self_py=agent.get_position()
        sensor_info=[0.]*self.num_sensor
        sensor_radius=agent.get_sensor_radius()
        
        for i in range(len(self.agents)):
            if i==idx:
                continue
            cur_agent=self.agents[i]
            if cur_agent==None:
                continue
            cur_agent_px,cur_agent_py=cur_agent.get_position()
            relative_px,relative_py=cur_agent_px-_self_px,cur_agent_py-_self_py
            update_s(sensor_info,relative_px,relative_py,self.num_sensor,self.agent_radius,sensor_radius)
        
        for i in range(len(self.obstacles)):
            cur_obstacle=self.obstacles[i]
            if cur_obstacle==None:
                continue
            cur_ob_px,cur_ob_py,cur_ob_radius=cur_obstacle.get_observable_state()
            relative_px,relative_py=cur_agent_px-_self_px,cur_agent_py-_self_py
            update_s(sensor_info,relative_px,relative_py,self.num_sensor,cur_ob_radius,sensor_radius)
        
        #传感器检测到与墙壁的距离
        epsilon=0.0000001
        for i in range(self.num_sensor):
            each_angle=2*math.pi/self.num_sensor
            cur_sensor_angle=each_angle*(i+1/2.0)+epsilon
            dist2_edge=sensor_radius
            if cur_sensor_angle>0 and cur_sensor_angle<=math.pi/2:
                x2edge,y2edge=self.x_max-_self_px,self.y_max-_self_py
                dist2_edge=min(x2edge/math.cos(cur_sensor_angle),y2edge/math.sin(cur_sensor_angle))
            elif cur_sensor_angle<=math.pi:
                x2edge,y2edge=_self_px,self.y_max-_self_py
                dist2_edge=min(x2edge/math.cos(math.pi-cur_sensor_angle),y2edge/math.cos(cur_sensor_angle-math.pi/2))
            elif cur_sensor_angle<=3/2.0*math.pi:
                x2edge,y2edge=_self_px,_self_py
                dist2_edge=min(x2edge/math.cos(cur_sensor_angle-math.pi),y2edge/math.cos(math.pi*3/2-cur_sensor_angle))
            else:
                x2edge,y2edge=self.x_max-_self_px,_self_py
                dist2_edge=min(x2edge/math.cos(2*math.pi-cur_sensor_angle),y2edge/math.cos(cur_sensor_angle-math.pi*3/2.0))
            dist2_edge=min(sensor_radius,dist2_edge)
            sensor_info[i]=(sensor_radius-dist2_edge)/sensor_radius
        return sensor_info
    
    def _get_state(self,agent_id):
        sensor_info=self._calc_sensor_info(agent_id)
        agent=self.agents[agent_id]
        _px,_py,_pgx,_pgy=agent.get_observable_state()
        _relative_x,_relative_y=_pgx-_px,_pgy-_py
        in_point = 1 if self.grab_counter[agent_id] > 0 else 0
        
        s=np.hstack([in_point,_relative_x,_relative_y,sensor_info])
        
        relative_xy=[_relative_x,_relative_y]#ToDo
        return s,relative_xy
        
        
    
class Viewer(pyglet.window.Window):
    color={'background':[1]*3+[1]}
    fps_display=pyglet.clock.ClockDisplay()
    
    def __init__(self,width,height,agents_info,obstacle_info):
        super(Viewer,self).__init__(width,height, resizable=False, caption='Agents', vsync=False)
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])
        
        self.agents_info=agents_info
        self.obstacles_info=obstacle_info
        
        self.viewer_xy=(width,height)
        
        self.center_coord = np.array((min(width, height)/2, ) * 2)
        self.batch = pyglet.graphics.Batch()
        self.batch_box=[]
        
    def render(self):
        pyglet.clock.tick()
        self._update_env()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
    
    def update(self,agents,obstacles=None):
        self.agents=agents
        if obstacles!=None:
            self.obstacles=obstacles
        
    def on_draw(self):
        self.clear()
        #self.batch.draw()
        #width=self.viewer_xy[0]
        #height=self.viewer_xy[1]
        #blank_cover_pts=[0,0,width,0,width,height,0,height]
        #c=(255,255,255)*4
        #blank_cover=pyglet.graphics.vertex_list(4, ('v2f', blank_cover_pts), ('c3B', c))
        #blank_cover.draw(pyglet.gl.GL_TRIANGLE_FAN)
        for graph in self.batch_box:
            graph.draw(pyglet.gl.GL_TRIANGLE_FAN)
        
    def _update_env(self):
        def generate_circle_verticles(px,py,radius,num_points=30):
            assert num_points>=4
            each_angle=2*math.pi/(num_points-1)
            verticle_list=[]
            for i in range(num_points):
                pt_x=px+radius*math.cos(each_angle*i)
                pt_y=py+radius*math.sin(each_angle*i)
                verticle_list+=pt_x,pt_y
            return verticle_list
        
        c1, c2, c3 = (249, 86, 86), (86, 109, 249), (149, 139, 65)
        self.batch_box=[]
        for agent in self.agents_info:
            _px,_py,_pgx,_pgy=agent.get_observable_state()
            #print("_px:",_px,"_py:",_py,"_pgx:",_pgx,"_pgy:",_pgy)
            _radius=agent.get_agent_radius()
            num_circle_point=30
            agent_verticles=generate_circle_verticles(_px,_py,_radius,num_circle_point)

            agent_circle = pyglet.graphics.vertex_list(num_circle_point, ('v2f', agent_verticles), ('c3B', c2*num_circle_point))
            self.batch_box.append(agent_circle)

            #self.batch.add(num_circle_point, pyglet.gl.GL_LINE_LOOP, None, ('v2f', agent_verticles), ('c3B', c2*num_circle_point))
            l_g=2
            goal_verticles=[_pgx-l_g,_pgy-l_g,_pgx+l_g,_pgy-l_g,_pgx+l_g,_pgy+l_g,_pgx-l_g,_pgy+l_g]
            goal_pt=pyglet.graphics.vertex_list(4, ('v2f', goal_verticles), ('c3B', c1*4))
            self.batch_box.append(goal_pt)
            #self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', goal_verticles), ('c3B', c1*4))
        for obstacle in self.obstacles_info:
            _px,_py,_radius=obstacle.get_observable_state()
            num_circle_point=30
            obstacle_verticles=generate_circle_verticles(_px,_py,_radius,num_circle_point)
            obstacle_circle = pyglet.graphics.vertex_list(num_circle_point, ('v2f', obstacle_verticles),('c3B', c3*num_circle_point))
            self.batch_box.append(obstacle_circle)
            #self.batch.add(num_circle_point, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', obstacle_verticles), ('c3B', c2*num_circle_point))
            #self.batch.add(num_circle_point, pyglet.gl.GL_QUADS, None, ('v2f', obstacle_verticles), ('c3B', c3*num_circle_point))
        
        
def test():
    num_agents=6
    num_obstacles=12
    agent_radius=10
    viewer_xy=(400,400)
    new_env=ENV(num_agents,num_obstacles,agent_radius,viewer_xy)
    new_env.reset()
    new_env.reset()
    new_env.render()
    new_env.render()
    #input()
    for i in range(200):
        actions=new_env.sample_action()
        new_env.step(actions)
        new_env.render()
    #new_env.render()
    
if __name__=="__main__":
    test()
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
                dist2_cur_agent=math.sqrt(pow(relative_px,2)+pow(relative_py,2))
                
                if dist2_cur_agent-self.agent_radius>sensor_radius:
                    continue
                if dist2_cur_agent<self.agent_radius:
                    sensor_info=[1.]*self.num_sensor
                    break
                center_angle_rad=math.atan2(relative_py,relative_px)
                split_angle=abs(math.atan2(self.agent_radius,math.sqrt(pow(dist2_cur_agent,2)-pow(self.agent_radius,2))))
                start_angle=center_angle_rad-split_angle
                end_angle=center_angle_rad+split_angle
                if start_angle<0:
                    start_angle+=2*math.pi
                if end_angle<0:
                    end_angle+=2*math.pi
                if center_angle_rad<0:
                    center_angle_rad+=2*math.pi
                each_angle=2*math.pi/self.num_sensor
                start_i=math.floor(start_angle/each_angle)
                end_i=math.ceil(end_angle/each_angle)
                center_i=center_angle_rad/each_angle
                for j in range(start_i,end_i-1):
                    if center_i>j and center_i<j+1:
                        sensor_info[j]=(sensor_radius-(dist2_cur_agent-self.agent_radius))/sensor_radius
                        continue
                    elif center_i>j:
                        line_j=j+1
                        length2_obstacle=math.cos(center_angle_rad-each_angle*line_j)-math.sqrt(self.agent_radius**2-(dist2_cur_agent*math.sin(center_angle_rad-each_angle*line_j))**2)
                        sensor_info[j]=(sensor_radius-length2_obstacle)/sensor_radius
                    elif center_i<j:
                        line_j=j
                        length2_obstacle=math.cos(center_angle_rad-each_angle*line_j)-math.sqrt(self.agent_radius**2-(dist2_cur_agent*math.sin(center_angle_rad-each_angle*line_j))**2)
                        sensor_info[j]=(sensor_radius-length2_obstacle)/sensor_radius
                        '''
    
    
    
    
    
    
    
    
    
    
    
    
    
