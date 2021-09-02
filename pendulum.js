var {Composite, Bodies, Body, Constraint} = Matter

class Pendulum{
    constructor(xx, yy, length, angle, group, brain) {
        this.score = 0;
        this.width = 150;
        this.height = 50;
        this.wheelSize = 20;
        this.wheelBase = this.wheelSize * 2.5;
        this.wheelAOffset = -this.wheelBase;
        this.wheelBOffset = this.wheelBase;
        this.wheelYOffset = 25;
        this.xx = xx;
        this.yy = yy;
        this.length = length;
        this.angle = angle;
        this.group = group;
        this.body = Composite.create();
        if (!brain) {
            this.brain = new NN(4,8,3)
        } else {
            this.brain = brain.copy()
        }
    }

    render() {
        const chassis = Bodies.rectangle(this.xx, this.yy, this.width, this.height, {
            collisionFilter: {
                group: this.group
            },
            mass:10,
            label:'chassis'
        });

        const arm = Bodies.rectangle(this.xx+Math.sin(this.angle)*this.length/2, this.yy-this.height/2-Math.cos(this.angle)*this.length/2, this.width*.1, this.length, {
            collisionFilter: {
                group: this.group
            },
            angle: this.angle,
            label:'arm',
            chamfer: {radius: this.width*.1*0.5}
        });

        const wheelA = Bodies.circle(this.xx + this.wheelAOffset, this.yy + this.wheelYOffset, this.wheelSize, { 
                collisionFilter: {
                    group: this.group
                },
                friction: 0.8,
                label:'wheelA'
            });

        const wheelB = Bodies.circle(this.xx + this.wheelBOffset, this.yy + this.wheelYOffset, this.wheelSize, { 
                collisionFilter: {
                    group: this.group
                },
                friction: 0.8,
                label:'wheelB'
            });

        const axelArm = Constraint.create({
            bodyA: arm,
            bodyB: chassis,
            pointA: { x: -Math.sin(this.angle)*this.length/2, y: Math.cos(this.angle)*this.length/2},
            pointB: { x: 0, y: -this.height/2},
            stiffness: 1,
            length: 0
        });

        const axelA = Constraint.create({
                bodyB: chassis,
                pointB: { x: this.wheelAOffset, y: this.wheelYOffset },
                bodyA: wheelA,
                stiffness: 1,
                length: 0
            });
                            
        const axelB = Constraint.create({
                bodyB: chassis,
                pointB: { x: this.wheelBOffset, y: this.wheelYOffset },
                bodyA: wheelB,
                stiffness: 1,
                length: 0
            });
            
        Composite.addBody(this.body, chassis);
        Composite.addBody(this.body, wheelA);
        Composite.addBody(this.body, wheelB);
        Composite.addBody(this.body, arm);
        Composite.addConstraint(this.body, axelArm);
        Composite.addConstraint(this.body, axelA);
        Composite.addConstraint(this.body, axelB);

        return this.body;
    }

    getChassis(){
        return this.body.bodies[0]
    }

    getArm(){
        return this.body.bodies[3]
    }

    moveRight(){
        Body.applyForce(this.getChassis(), this.getChassis().position, { x:0.01, y:0 });
    }

    moveLeft(){
        Body.applyForce(this.getChassis(), this.getChassis().position, { x:-0.01, y:0 });
    }

    moveStop(){
        Body.applyForce(this.getChassis(), this.getChassis().position, { x:0, y:0 });
    }

    think(){
        tf.tidy(() => {
            let inputs = [];
            inputs[0] = this.getChassis().position.x/displayWidth
            inputs[1] = this.getChassis().speed
            inputs[2] = this.getArm().angle/Math.PI
            inputs[3] = this.getArm().angularVelocity/0.03411361568115584
            
            let y = tf.argMax(this.brain.predict(inputs)).arraySync();
            if (y == 0) {this.moveLeft()} else if (y == 1) {this.moveRight()} else if (y == 2) {this.moveStop()}
        });
    }

    score(){
        this.score++
    }
}