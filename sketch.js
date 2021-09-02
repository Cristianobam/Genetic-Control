var {Engine, Render, Runner, World, Bodies, Events, Body} = Matter;

let displayWidth = 800,
    displayHeight = 700,
    group = Body.nextGroup(true),
    force = .01,
    engine = Engine.create(),
    world = engine.world;

let render = Render.create({
    element: document.body,
    engine: engine,
    options: {
        width: displayWidth,
        height: displayHeight,
        showAngleIndicator: false,
        showCollisions: true,
        wireframes: false
    }
});

Render.run(render);

let runner = Runner.create();
Runner.run(runner, engine);


let ground = Bodies.rectangle(displayWidth/2, displayHeight-25, displayWidth, 50, {isStatic: true}), // Ground
    wallLeft = Bodies.rectangle(0, displayHeight/2, 50, displayHeight, {isStatic: true, friction:1}), // Wall Left
    wallRight = Bodies.rectangle(displayWidth, displayHeight/2, 50, displayHeight, {isStatic: true, friction:1}) // Wall Right

World.add(world, [ground, wallLeft, wallRight]);


// Initialize Entities
const nCars = 50
let cars = []
for (let i = 0; i < nCars; i++){
    cars.push(new Pendulum(displayWidth/2, displayHeight-95, 300, 0, group)); // New pendulum object
}

cars.forEach((car)=>{World.add(world, car.render())}); // Add to the world

//looks for key presses and logs them
var keys = [];
document.body.addEventListener("keydown", function(e) {
    keys[e.key] = true;
});

document.body.addEventListener("keyup", function(e) {
    keys[e.key] = false;
});


Events.on(engine, "beforeTick", function(event) {
    cars.forEach((car)=> { car.think() })

    for (let i = cars.length-1; i >= 0; i--){
        if ((Matter.SAT.collides(cars[i].getArm(), wallLeft).collided) || (Matter.SAT.collides(cars[i].getArm(), wallRight).collided) || (Matter.SAT.collides(cars[i].getArm(), ground).collided)){
            World.remove(world, cars[i].render())
            cars.splice(i, 1);
        }
    }

    if (keys['ArrowRight']){
        cars[0].moveRight()
    } else if (keys['ArrowLeft']){
        cars[0].moveLeft()
    }
});