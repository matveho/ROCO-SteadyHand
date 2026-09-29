cd ~/ROCO-SteadyHand-live
export ROBOT_NAME=dm/vgfcb66075ea-1u

pkill -f 'dexsensor.*head_camera' || true

nohup dexsensor -v info launch \
  --config /etc/dexmate/dexsensor/default.toml \
  --robot dm/vgfcb66075ea-1u \
  --sensor head_camera \
  > ~/head_camera.log 2>&1 &

sleep 10
tail -n 80 ~/head_camera.log