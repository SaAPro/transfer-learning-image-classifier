//// To be defined by the user : //////////////////////////////////////////////
const numberImagesToRead = 5;
const trainNoAction = 15;
const knn = 3;
///////////////////////////////////////////////////////////////////////////////

const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');

let net;

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}

async function app() {
  console.log('Loading mobilenet..');
	document.getElementById('console').innerText = 'Loading mobilenet, plase wait..';

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

	// Setup the webcam
	document.getElementById('console').innerText = 'Setting webcam..';
  await setupWebcam();

  // Reads several images from the webcam and associates it with a specific class
  // index.
  const addExample = (classId, iter) => {

		// Add a specific number of images for a certain class
		for (var i = 0; i < iter; i++) {
			// Get the intermediate activation of MobileNet 'conv_preds' and pass that
			// to the KNN classifier.
			const activation = net.infer(webcamElement, 'conv_preds');
			// Pass the intermediate activation to the classifier.
			classifier.addExample(activation, classId);
		}

		document.getElementById('console').innerText = 'Images added to classification';
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample('class-A', numberImagesToRead));
  document.getElementById('class-b').addEventListener('click', () => addExample('class-B', numberImagesToRead));
  document.getElementById('class-c').addEventListener('click', () => addExample('class-C', numberImagesToRead));

	// Train 'no-action' class
	document.getElementById('console').innerText = 'Training no-action class..';
	await tf.nextFrame(); // to be sure webcam is started
	addExample('no-action', trainNoAction)

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation, k = knn);

      document.getElementById('console').innerText = `prediction: ${result.label}
      probability: ${result.confidences[result.label]}`;
    }
    await tf.nextFrame();
  }
}

app();
