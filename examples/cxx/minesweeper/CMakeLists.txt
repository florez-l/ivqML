## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_SOURCE_DIR}/lib/cxx
  ${PROJECT_BINARY_DIR}/lib/cxx
  )

add_executable(
  MineSweeper
  CPUMineSweeperPlayer.cxx
  CPUMineSweeperPlayer.h
  HumanMineSweeperPlayer.cxx
  HumanMineSweeperPlayer.h
  LogisticMineSweeperPlayer.cxx
  LogisticMineSweeperPlayer.h
  MineSweeperBoard.cxx
  MineSweeperBoard.h
  MineSweeperPlayerBase.cxx
  MineSweeperPlayerBase.h
  play.cxx
  RandomMineSweeperPlayer.cxx
  RandomMineSweeperPlayer.h
  )
target_link_libraries(MineSweeper Eigen3::Eigen)

## eof - $RCSfile$
