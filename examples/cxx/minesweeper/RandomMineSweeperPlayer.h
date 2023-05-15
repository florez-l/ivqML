// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __RandomMineSweeperPlayer__h__
#define __RandomMineSweeperPlayer__h__

#include "MineSweeperPlayerBase.h"
#include <utility>
#include <vector>

/**
 */
class RandomMineSweeperPlayer
  : public MineSweeperPlayerBase
{
public:
  RandomMineSweeperPlayer( );
  virtual ~RandomMineSweeperPlayer( ) = default;

  virtual void configure(
    unsigned int w, unsigned int h, unsigned long n
    ) override;
  virtual void play( unsigned int& i, unsigned int& j ) override;

protected:
  std::vector< std::pair< unsigned int, unsigned int > > m_Options;
};

#endif // __RandomMineSweeperPlayer__h__

// eof - $RCSfile$
