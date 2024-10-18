vidReader = VideoReader('Demo_008.mp4');

opticFlow = opticalFlowHS;
h = figure;
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);

flow_x=nan(vidReader.Width,vidReader.Height,vidReader.NumFrames);
flow_y=nan(vidReader.Width,vidReader.Height,vidReader.NumFrames);
flow_ori=nan(vidReader.Width,vidReader.Height,vidReader.NumFrames);
flow_mag=nan(vidReader.Width,vidReader.Height,vidReader.NumFrames);
k=1;
while hasFrame(vidReader)
    frameRGB = readFrame(vidReader);
    frameGray = im2gray(frameRGB);  
    flow = estimateFlow(opticFlow,frameGray);
    flow_x(:,:,k)=flow.Vx;
    flow_y(:,:,k)=flow.Vy;
    flow_ori(:,:,k)=flow.Orientation;
    flow_mag(:,:,k)=flow.Magnitude;
    imshow(frameRGB)
    hold on
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',60,'Parent',hPlot);
    hold off
    pause(10^-3)
    k=k+1;
end