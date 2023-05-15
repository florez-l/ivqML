// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __HumanMineSweeperPlayer__h__
#define __HumanMineSweeperPlayer__h__

#include "MineSweeperPlayerBase.h"

/**
 */
class HumanMineSweeperPlayer
  : public MineSweeperPlayerBase
{
public:
  HumanMineSweeperPlayer( );
  virtual ~HumanMineSweeperPlayer( ) = default;

  virtual void play( unsigned int& i, unsigned int& j ) override;
};

#endif // __HumanMineSweeperPlayer__h__

// eof - $RCSfile$
