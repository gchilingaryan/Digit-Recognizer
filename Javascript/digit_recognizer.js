function setup() {
  createCanvas(450, 400);
  background(0);
}

function draw() {
  if (mouseIsPressed) {
    strokeWeight(20);
    stroke(255);
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}

function erasePressed() {
  background(0);
}

async function guessPressed() {
  let pixels = [];
  let img = get();
  img.resize(28, 28);
  img.loadPixels();
  for (let i = 0; i < 784; i++) {
    let each_pixel = img.pixels[i * 4];
    pixels[i] = each_pixel;
  }
  const model = await tf.loadModel('http://127.0.0.1:8080/model.json');
  let prediction = model.predict(tf.tensor(pixels).reshape([-1,28,28,1]));
  prediction.print();
  let predicted = prediction.flatten();
  predicted.argMax().print();
}
