document.addEventListener('alpine:init', function () {
    Alpine.data('data', function () {
        return {
            // Your code goes here
            prediction: { 'confidence': 0.0, 'predictedLabel': '' },
            baseUrl: 'http://24.199.98.176',
            isPreview: false,
            isPredicted: false,
            isPredictBtn: false,
            isLoading: false,

            getIsPreviewState() {
                return this.isPreview;
            },

            handlePredictSubmit(e) {
                e.preventDefault();
                // preparing form data to be sent with the post request.
                const formData = new FormData();

                // Uploading the image to be added in the form data.
                const uploadedImg = document.querySelector('#file');

                // adding the image in the form using the append method from the form data
                formData.append(
                    'file',
                    uploadedImg.files[0]
                );

                // Sending request with the form data included
                axios.post(`${this.baseUrl}/ml/predict`, formData).then(res => {
                    const { confidence, predicted_label } = res.data.prediction;
                    this.isPreview = false;
                    this.isPredicted = true;

                    setTimeout(() => {
                        this.displayFile(uploadedImg);
                        this.prediction = {
                            confidence: confidence,
                            predictedLabel: predicted_label
                        }
                    }, 1000);
                })
            },

            loadFile(e) {
                this.isPreview = true;
                this.displayFile(e);
                setTimeout(() => {
                    this.isPredictBtn = true;
                }, 2000)
            },

            displayFile(e) {
                const previewTimer = setTimeout(() => {
                    let image = document.getElementById('output');
                    image.src = URL.createObjectURL(e.target == null ? e.files[0] : e.target.files[0]);
                }, 1000);
            }
        }
    })
})