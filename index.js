async function run_hw3_regression() {    
    const MEANS = [14.117635164835171, 19.18503296703298, 91.88224175824185, 654.3775824175825, 0.09574402197802204, 0.10361931868131863, 0.08889814505494498, 0.04827987032967031, 0.18109868131868148, 0.06275676923076925, 0.40201582417582393, 1.2026868131868136, 2.858253406593405, 40.0712989010989, 0.00698907472527473, 0.025635448351648396, 0.0328236723076923, 0.011893940659340657, 0.020573512087912114, 0.003820455604395603, 16.23510329670329, 25.535692307692308, 107.10312087912091, 876.9870329670341, 0.13153213186813184, 0.2527418021978023, 0.27459456923076936, 0.11418222197802197, 0.29050219780219777, 0.0838678461538462];
    const SCALES = [3.5319276091287684, 4.261314035201523, 24.29528446596607, 354.5529252060648, 0.013907698124434402, 0.052412805496132024, 0.07938050908411763, 0.038018354057687886, 0.027457084964442154, 0.0072017850581413915, 0.2828495575198162, 0.5411516758817481, 2.068931392290445, 47.18438200914984, 0.003053473706769491, 0.01858629695791424, 0.032110245434099904, 0.006287187209688091, 0.008162966415892984, 0.0027840687418581585, 4.805977154451531, 6.058439641882756, 33.33796863783808, 567.0486811155924, 0.02305712569565531, 0.15484384737160206, 0.20916786137677873, 0.06525425828147159, 0.06308179580673515, 0.017828276003334045];
    const N_FEATURES = 11;

    try {
        const session = await ort.InferenceSession.create('./regression_model.onnx');
        const inputString = document.getElementById("reg_inputs").value;
        
        // 2. Process inputs
        const rawFeatures = inputString.split(',').map(Number);
        if (rawFeatures.length !== N_FEATURES) {
            alert(`Error: Expected ${N_FEATURES} features, got ${rawFeatures.length}`);
            return;
        }
        
const scaledFeatures = rawFeatures.map((val, i) => {
    const mean = MEANS[i];
    const scale = SCALES[i];
    
    if (scale === 0) {
        return 0; // If variance was 0, the scaled feature is 0
    }
    
    return (val - mean) / scale;
});
        const data = Float3TensorsorArray.from(scaledFeatures);
        const tensor = new ort.Tensor('float32', data, [1, N_FEATURES]);

        // 3. Run model and show output
        // This 'input' name MUST match your Python export name
        const feeds = { 'input': tensor }; 
        const results = await session.run(feeds);
        
        // This 'output' name MUST match your Python export name
        const prediction = results.output.data[0]; 
        document.getElementById("reg_output").innerHTML = prediction.toFixed(3);

    } catch (e) { console.error(e); }
}


async function run_hw3_classification() {
   
    const MEANS = [14.117635164835171, 19.18503296703298, 91.88224175824185, 654.3775824175825, 0.09574402197802204, 0.10361931868131863, 0.08889814505494498, 0.04827987032967031, 0.18109868131868148, 0.06275676923076925, 0.40201582417582393, 1.2026868131868136, 2.858253406593405, 40.0712989010989, 0.00698907472527473, 0.025635448351648396, 0.0328236723076923, 0.011893940659340657, 0.020573512087912114, 0.003820455604395603, 16.23510329670329, 25.535692307692308, 107.10312087912091, 876.9870329670341, 0.13153213186813184, 0.2527418021978023, 0.27459456923076936, 0.11418222197802197, 0.29050219780219777, 0.0838678461538462]
    const SCALES = [3.5319276091287684, 4.261314035201523, 24.29528446596607, 354.5529252060648, 0.013907698124434402, 0.052412805496132024, 0.07938050908411763, 0.038018354057687886, 0.027457084964442154, 0.0072017850581413915, 0.2828495575198162, 0.5411516758817481, 2.068931392290445, 47.18438200914984, 0.003053473706769491, 0.01858629695791424, 0.032110245434099904, 0.006287187209688091, 0.008162966415892984, 0.0027840687418581585, 4.805977154451531, 6.058439641882756, 33.33796863783808, 567.0486811155924, 0.02305712569565531, 0.15484384737160206, 0.20916786137677873, 0.06525425828147159, 0.06308179580673515, 0.017828276003334045];
    const N_FEATURES = 30;

    try {
        const session = await ort.InferenceSession.create('./classification_model.onnx');
        const inputString = document.getElementById("cls_inputs").value;

       
        const rawFeatures = inputString.split(',').map(Number);
        if (rawFeatures.length !== N_FEATURES) {
            alert(`Error: Expected ${N_FEATURES} features, got ${rawFeatures.length}`);
            return;
        }

const scaledFeatures = rawFeatures.map((val, i) => {
    const mean = MEANS[i];
    const scale = SCALES[i];
    
    if (scale === 0) {
        return 0; // If variance was 0, the scaled feature is 0
    }
    
    return (val - mean) / scale;
});

        const data = Float32Array.from(scaledFeatures);
        const tensor = new ort.Tensor('float32', data, [1, N_FEATURES]);

        // 3. Run model and show output
        const feeds = { 'input': tensor }; 
        const results = await session.run(feeds);
        
        // Convert logit -> probability -> 0/1
        const logit = results.output.data[0];
        const probability = 1.0 / (1.0 + Math.exp(-logit)); // Sigmoid function
        const prediction = (probability > 0.5) ? 1 : 0;
        
        document.getElementById("cls_output").innerHTML = `<b>${prediction}</b> (Prob: ${probability.toFixed(2)})`;

    } catch (e) { console.error(e); }
}
