// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Functions__h__
#define __ivqML__Model__Functions__h__

#include <ivqML/Config.h>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _TReal >
    class Functions
    {
    public:
      using Self      = Functions;
      using TReal     = _TReal;
      using TMatrix   = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
      using TMap = Eigen::Map< TMatrix >;
      using TElementwise = std::function< TReal( const TReal&, bool ) >;
      using TFunction = std::function< void( TMap&, const TMap&, bool ) >;
      using TPair = std::pair< std::string, TFunction >;

    public:
      static TPair Get( const std::string& name );
    };
  } // end namespace
} // end namespace

#endif // __ivqML__Model__Functions__h__

// eof - $RCSfile$
