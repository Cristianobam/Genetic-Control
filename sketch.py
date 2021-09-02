import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
import json

width, height = (800, 700)

pygame.init()
display = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 34)

def update_fps():
	fps = str(int(clock.get_fps()))
	fps_text = font.render(fps, 1, pygame.Color("coral"))
	return fps_text

space = pymunk.Space()
space.gravity = 0, 980
FPS = 60

POP_SIZE = 102

draw_options = pymunk.pygame_util.DrawOptions(display)
draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
draw_options.flags |= pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS
draw_options.constraint_color = (255,255,255,255)

collision_types = {
    'wall': 1,
    'car': 4
}

class Engine:
    def __init__(self) -> None:
        pass
    
    def color(self):
        if not self.isStatic:
            color = np.random.choice(['#556270', '#4ECDC4', '#C7F464', '#FF6B6B', '#C44D58'])
        else:
            color = '#2e2b44'
        color = self.hex2rgb(color)
        color.append(255)
        return color
        
    def convert_coordinates(self, position):
        return int(position[0]), int(position[1])
    
    def hex2rgb(self, hexcode):
        hexcode = hexcode.lstrip('#')
        return [int(hexcode[i:i+2], 16) for i in (0, 2, 4)]

class Ground(Engine):
    def __init__(self) -> None:
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.thickness = 20
        self.body.position = width/2, height-self.thickness/2
        self.shape = pymunk.Poly.create_box(self.body, (width, self.thickness))
        self.shape.collision_type = collision_types['wall']
        self.shape.elasticity = 1
        self.shape.friction = 1
        self.isStatic = True
        self.shape.color = self.color()
        self.shape.filter = pymunk.ShapeFilter(categories=0b10, mask=0b01)
        space.add(self.body, self.shape)
        
class Wall(Engine):
    def __init__(self, x, y, width, height) -> None:
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = x, y
        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.collision_type = collision_types['wall']
        self.shape.elasticity = 1
        self.shape.friction = 1
        self.isStatic = True
        self.shape.color = self.color()
        self.shape.filter = pymunk.ShapeFilter(categories=0b10, mask=0b01)
        space.add(self.body, self.shape)

class Car(Engine):
    def __init__(self, x, y, length, angle, group, brain=None) -> None:
        super().__init__()
        self.isStatic = False
        self.width = 150
        self.height = 50
        self.wheelSize = 20
        self.wheelBase = self.wheelSize * 2.5
        self.wheelAOffset = -self.wheelBase
        self.wheelBOffset = self.wheelBase
        self.wheelYOffset = 25
        self.vertex = np.array([(-self.width/2, -self.height/2), (self.width/2, -self.height/2),
                        (self.width/2, self.height/2), (-self.width/2, self.height/2)])
        self.brain = Brain(4,5,2) if brain is None else brain
        self.fitness = 0
        self.avaiable = True
        self.speed = 0
        
        self.chassis = pymunk.Body()
        self.chassis.position = x, y
        self.chassis_shape = pymunk.Poly(self.chassis, self.vertex.tolist())
        self.chassis_shape.collision_type = collision_types['car']
        self.chassis_shape.friction = 1
        self.chassis_shape.elasticity = .3
        self.chassis_shape.mass = 10
        self.chassis_shape.color = self.color()
        self.chassis_shape.filter = pymunk.ShapeFilter(group=group, categories=0b01, mask=0b10)

        self.wheelA = pymunk.Body()
        self.wheelA.position = x+self.wheelAOffset, y+self.wheelYOffset
        self.wheelA_shape = pymunk.Circle(self.wheelA, self.wheelSize)
        self.wheelA_shape.friction = 1
        self.wheelA_shape.elasticity = .3
        self.wheelA_shape.mass = 1
        self.wheelA_shape.color = self.color()
        self.wheelA_shape.filter = pymunk.ShapeFilter(group=group, categories=0b01, mask=0b10)

        self.wheelB = pymunk.Body()
        self.wheelB.position = x+self.wheelBOffset, y+self.wheelYOffset
        self.wheelB_shape = pymunk.Circle(self.wheelB, self.wheelSize)
        self.wheelB_shape.friction = 1
        self.wheelB_shape.elasticity = .3
        self.wheelB_shape.mass = 1
        self.wheelB_shape.color = self.color()
        self.wheelB_shape.filter = pymunk.ShapeFilter(group=group, categories=0b01, mask=0b10)

        self.arm = pymunk.Body()
        self.arm.position = x, y - self.height/2
        self.arm_shape = pymunk.Segment(self.arm, (0,0),
                                    (length*np.sin(angle),
                                    -length*np.cos(angle)), 5)
        
        self.arm_shape.collision_type = collision_types['car']
        self.arm_shape.mass = 1
        self.arm_shape.color = self.color()
        self.arm_shape.filter = pymunk.ShapeFilter(group=group, categories=0b01, mask=0b10)
        
        space.add(self.wheelA, self.wheelA_shape)
        space.add(self.wheelB, self.wheelB_shape)
        space.add(self.arm, self.arm_shape)
        space.add(self.chassis, self.chassis_shape)
        
        self.jointA = pymunk.constraints.PivotJoint(self.wheelA, self.chassis,
                                            (0, 0),
                                            (self.wheelAOffset, self.wheelYOffset))
        self.jointA.collide_bodies = True
        space.add(self.jointA)
        
        self.jointB = pymunk.constraints.PivotJoint(self.wheelB, self.chassis,
                                            (0, 0),
                                            (self.wheelBOffset, self.wheelYOffset))
        self.jointB.collide_bodies = True
        space.add(self.jointB)
        
        self.jointArm = pymunk.constraints.PivotJoint(self.arm, self.chassis,
                                            (0, 0),
                                            (0, -self.height/2))
        self.jointArm.collide_bodies = True
        space.add(self.jointArm)
        
        self.motorA = pymunk.SimpleMotor(self.wheelA, self.chassis, self.speed)
        self.motorB = pymunk.SimpleMotor(self.wheelB, self.chassis, self.speed)
        
        space.add(self.motorA, self.motorB)

    def moveLeft(self):
        #self.chassis.apply_force_at_local_point(pymunk.Vec2d(-2.5E3,0), (0,0))
        self.motorA.rate -= 1
        self.motorB.rate -= 1
    def moveRight(self):
        #self.chassis.apply_force_at_local_point(pymunk.Vec2d(2.5E3,0), (0,0))
        self.motorA.rate += 1
        self.motorB.rate += 1

    def remove(self):
        self.avaiable = False
        space.remove(self.chassis, self.arm, self.wheelA, self.wheelB)
        space.remove(self.chassis_shape, self.arm_shape, self.wheelA_shape, self.wheelB_shape)
        space.remove(self.jointA, self.jointB, self.jointArm)
        space.remove(self.motorA, self.motorB)

    def think(self):
        input = [self.arm.angular_velocity/(2*np.pi),
                self.arm.angle/(2*np.pi),
                self.chassis.position[0]/width,
                self.chassis.velocity[0]/width]
        ans = np.argmax(self.brain.predict(np.reshape(input, (-1,4)))[0])
        if ans==0:
            self.moveLeft()
        elif ans==1:
            self.moveRight()
        elif ans==2:
            pass

    def map(self, n, start1, stop1, start2, stop2):
        return (n - start1) / (stop1 - start1) * (stop2 - start2) + start2

    def scoreAdd(self):
        self.fitness +=  self.map(np.abs(self.arm.angle), 0, np.pi, 2, 0)
        self.fitness +=  self.map(np.abs(self.chassis.position[0]-width/2), 0, width/2, 1, 0)
        self.fitness -= self.map(abs(self.arm.angular_velocity), 0, 2*np.pi, 0, 2)

class Brain():
    def __init__(self, a, b, c, d=None):
        if isinstance(a, NN):
            self.model = a
            self.input = b
            self.hidden = c
            self.output = d
        else:
            self.input = a
            self.hidden = b
            self.output = c
            self.model = self.build()

    def build(self):
        return NN(self.input, self.hidden, self.output)
    
    def predict(self, inputs):
        return self.model.forward(inputs)
    
    def copy(self):
        modelCopy = self.build()
        oldWeight = self.model.get_weights()
        modelCopy.set_weights(oldWeight)
        return Brain(modelCopy, self.input, self.hidden, self.output)
    
    def mutate(self, rate):
        oldWeights = self.model.get_weights()
        mutatedWeights = []
        for oldWeight in oldWeights:
            shape = oldWeight.shape
            values = oldWeight.flatten()
            for i in range(len(values)):
                if np.random.rand() < rate:
                    values[i] += np.random.randn()*.5
            mutatedWeights.append(np.reshape(values, shape))
        self.model.set_weights(mutatedWeights)
        
    def crossOver(self, parent):
        p1Weights = self.model.get_weights()
        p2Weights = parent.model.get_weights()
        offspringWeights = []
        for weights in zip(p1Weights, p2Weights):
            shape = weights[0].shape
            values1 = weights[0].flatten()
            values2 = weights[1].flatten()
            values = []
            for i in range(len(values1)):
                if i % 2 == 0:
                    values.append(values1[i])
                else:
                    values.append(values2[i])
            offspringWeights.append(np.reshape(values, shape))
        offspring = Brain(self.input, self.hidden, self.output)
        offspring.model.set_weights(offspringWeights)
        return offspring

class NN():
    def __init__(self, input, hidden, output):
        self.inputLayer = input
        self.hiddenLayer = hidden
        self.outputLayer = output
        
        self.W1 = np.random.randn(self.inputLayer, self.hiddenLayer)
        self.B1 = np.zeros(self.hiddenLayer)
        self.W2 = np.random.randn(self.hiddenLayer, self.outputLayer)
        self.B2 = np.ones(self.outputLayer)
        
    def get_weights(self):
        return [self.W1, self.B1, self.W2, self.B2]

    def set_weights(self, weights):
        self.W1 = weights[0]
        self.B1 = weights[1]
        self.W2 = weights[2]
        self.B2 = weights[3]

    def forward(self, X):
        self.z1 = self.sigmoid(np.dot(np.array(X), self.W1) + self.B1)
        self.z2 = self.sigmoid(np.dot(self.z1, self.W2) + self.B2)
        return self.softmax(self.z2)

    def softmax(self, z):
        return np.exp(np.array(z))/np.sum(np.exp(np.array(z)))

    def sigmoid(self, z):
        return 1./(1+np.exp(-np.array(z)))  

class GA():
    def __init__(self):
        self.genNum = 0
        self.generation = {}
    
    def getFitness(self):
        return [subject.fitness for subject in self.population]
    
    def getFittest(self, n=2):
        fitness = self.getFitness()
        fittest = np.sort(fitness)[-n:]
        return [fitness.index(i) for i in fittest]
        
    def crossOver(self, n=2):
        parents = [self.population[index] for index in self.getFittest(n=n)]
        combinations = [(parents[i], parents[j]) for i in range(len(parents)) for j in range(i,len(parents))]
        offsprings = []
        for pair in combinations:
            pair[0].brain.crossOver(pair[1].brain)
            offsprings.append(pair[0].brain.crossOver(pair[1].brain))
        return offsprings

    def newGeneration(self, population, n=2):
        self.population = population
        self.popSize = len(population)
        self.save()
        newGeneration = []
        offsprings = self.crossOver(n)
        index = int(self.popSize/len(offsprings))
        for n in range(len(offsprings)):
            for i in range(index):
                offspring = offsprings[n]
                offspring.mutate(.3)
                newGeneration.append(offspring)
        return newGeneration  
        
    def save(self):
        self.generation[self.genNum] = {}
        self.generation[self.genNum]['fitness'] = np.sort(self.getFitness())[-1]
        self.generation[self.genNum]['fittest'] = [self.population[index] for index in self.getFittest(n=1)][0].brain.model.get_weights()
        self.generation[self.genNum]['fittest'] = [l.tolist() for l in self.generation[self.genNum]['fittest']]
        print(f'--- Generation {self.genNum} ---')
        print(self.generation[self.genNum]['fittest'])
        
        with open('data.json', 'w') as f:
            json.dump(self.generation, f)
        self.genNum += 1
        
    def save_model(self, model):
        generation = {}
        generation['fitness'] = model.fitness
        generation['fittest'] = model.brain.model.get_weights()
        generation['fittest'] = [l.tolist() for l in generation['fittest']]
        with open(f'model_saved_#{self.genNum}.json', 'w') as f:
            json.dump(generation, f)

def game():
    GEN = 0
    Ground()
    Wall(10, height/2, 20, height)
    Wall(width-10, height/2, 20, height)
    cars = {i+1:Car(width/2, height*.9, 200, 0, i+1) for i in range(POP_SIZE)}
    ga = GA()
    generation = []

    def remove(arbiter, space, data):
        index = arbiter.shapes[-1].filter.group
        cars[index].remove()
        generation.append(cars[index])
        del cars[index]
        return True

    h = space.add_collision_handler(collision_types['wall'], collision_types['car'])
    h.begin = remove

    timeInit = pygame.time.get_ticks()/1000
    while True:
        keys_pressed = pygame.key.get_pressed()

        if keys_pressed[pygame.K_LEFT]:
            pass

        elif keys_pressed[pygame.K_RIGHT]:
            pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    #list(cars.values())[-1].arm.apply_force_at_local_point(pymunk.Vec2d(-1E3,0), (0,0))
                    list(cars.values())[-1].motorA.rate = -5
                    list(cars.values())[-1].motorB.rate = -5
                if event.key == pygame.K_RIGHT:
                    #list(cars.values())[-1].arm.apply_force_at_local_point(pymunk.Vec2d(1E3,0), (0,0))
                    list(cars.values())[-1].motorA.rate = 5
                    list(cars.values())[-1].motorB.rate = 5
                if event.key == pygame.K_s:
                    ga.save_model(list(cars.values())[-1])
                    
                elif event.key == pygame.K_p:
                    pygame.image.save(display, "image.png")

        for b in space.bodies:
            p = pymunk.pygame_util.to_pygame(b.position, display)

        display.fill((24,24,29))
        clock.tick(FPS)
        space.step(1./FPS)
        space.debug_draw(draw_options)
        display.blit(update_fps(), (0,0))
        display.blit(font.render(f'#{GEN}', 1, pygame.Color("coral")), (0,40))
        timeNow = pygame.time.get_ticks()/1000 - timeInit
        display.blit(font.render(f'Secs: {timeNow:.3f}', 1, pygame.Color("coral")), (0,80))
        threshold = 3 + 100*np.exp(-GEN/120)
        display.blit(font.render(f'Thresh: {threshold:.3f}', 1, pygame.Color("coral")), (0,120))
        pygame.display.update()
        
        for car in cars.values():
            car.think()
            car.scoreAdd()
        
        if len(cars.values()) == 0 or timeNow >= threshold:
            keys = list(cars.keys()).copy()
            for key in keys:
                cars[key].remove()
                generation.append(cars[key])
                del cars[key]
            
            cars = {n+1:Car(width/2, height*.9, 200, 0, n+1, i) for n,i in enumerate(ga.newGeneration(generation, n=2))}
            generation = []
            GEN += 1
            timeInit = pygame.time.get_ticks()/1000
game()
pygame.quit()